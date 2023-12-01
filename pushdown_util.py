# utils for running pushdown layers in various settings
import torch
import collate
import layers
import models
import copy
import torch.nn.functional as F
from tqdm import tqdm
from interfaces import add_eos

from data_utils.text_helpers import compute_attachment_labels_text, compute_stack_tape
import numpy as np
from nltk import Tree


def compute_stack_logprobs(lm, attachment_logits, stack_labels, input_lens):
    ### attachment_logits: b x token x token_to_reduce_with

    log_probs = []
    is_synchronous = lm.synchronous
    for bs, logit_set in enumerate(attachment_logits):
        curr_len = input_lens[bs]
        log_probs_curr = []
        logit_set = logit_set[:curr_len, :curr_len]
        for idx, token_logit in enumerate(logit_set):
            if is_synchronous and idx == curr_len - 1:
                ### if we are a synchronous transformer, we don't care what the reduce corresponding to the last prediction is
                ### because we always predict </s> at the end, and we dont care how it attaches.
                continue

            if is_synchronous:
                logits_considered = token_logit[: idx + 2]
            else:
                logits_considered = token_logit[: idx + 1]
            log_probs_curr.append(
                F.log_softmax(logits_considered, dim=0)[stack_labels[bs][idx]].item()
            )
        log_probs.append(log_probs_curr)
    return log_probs


def compute_per_token_logprob(lm, str_logits, inputs, input_lens):
    str_logprobs = []
    # (bs x len x vocab)
    str_logits = str_logits.transpose(0, 1)
    eos_token = torch.tensor([lm.encoder_eos]).to(inputs.device)
    for idx, (c_input, str_logprob) in enumerate(zip(inputs, str_logits)):
        curr_len = input_lens[idx]
        ## len x vocab
        ### shift input by 1 to evaluate LM
        target = torch.cat([c_input[1:curr_len], eos_token])
        eos_removed_logits = str_logprob[:curr_len]
        eos_logprobs = F.log_softmax(eos_removed_logits, dim=1)
        logprobs_curr = torch.gather(eos_logprobs, 1, target.unsqueeze(1)).squeeze(1)
        str_logprobs.append(logprobs_curr.cpu().numpy())
    return str_logprobs


def tokenizer_helper(
    lm,
    tokenizer,
    data_collator,
    inp_slice,
    parse_slice_or_labels=None,
    stack_type_info=None,
):
    inp_list = [tokenizer(s) for s in inp_slice]
    in_lens = [len(s) for s in inp_list]
    synchronous = lm.synchronous
    if parse_slice_or_labels is not None:
        stack_out = [
            compute_attachment_labels_text(parse, sent)
            for parse, sent in zip(parse_slice_or_labels, inp_slice)
        ]

        if synchronous:
            stack_labels = [out["attachment_labels"][1:] for out in stack_out]
        else:
            stack_labels = [out["attachment_labels"] for out in stack_out]
        if stack_type_info == "depth":
            stack_tapes = [
                compute_stack_tape(
                    out["attachment_labels"],
                    out["cstart_info"],
                    with_depth_info=True,
                    synchronous=synchronous,
                )
                for idx, out in enumerate(stack_out)
            ]
        else:
            stack_tapes = [
                compute_stack_tape(
                    out["attachment_labels"],
                    out["cstart_info"],
                    None,
                    synchronous=synchronous,
                )
                for idx, out in enumerate(stack_out)
            ]
        inp_to_collate = [
            {"in": x, "in_len": y, "stack_tape": stack, "stack_label": stack_label}
            for x, y, stack, stack_label in zip(
                inp_list, in_lens, stack_tapes, stack_labels
            )
        ]
        inp = data_collator(inp_to_collate)
        in_len = inp["in_len"].long()
        return (
            inp["in"].transpose(0, 1),
            in_len,
            inp["stack_label"].transpose(0, 1),
            inp["stack_tape"].transpose(0, 1),
        )
    else:
        inp_to_collate = [{"in": x, "in_len": y} for x, y in zip(inp_list, in_lens)]
        inp = data_collator(inp_to_collate)
        in_len = inp["in_len"].long()
        return (
            inp["in"].transpose(0, 1),
            in_len,
        )


@torch.no_grad()
def make_preds_with_greedy_decoding(
    lm,
    tokenizer,
    sents,
    gpu_id=0,
    do_sample=False,
    silent=False,
    get_final_answer=False,
):
    """
    Use the language model to make predictions on the given sentences.
    But, use greedy decoding to infer the parse structure.
    Also, if do_sample is True, then use sampling instead of greedy decoding.

    Output:
        - logprob of each sentence
        - stack_info: list of stack_info for each sentence
    """
    data_collator = collate.VarLengthCollate(None)
    batch_size = 32
    st = 0
    device = torch.device("cuda:{}".format(gpu_id))
    all_attachment_decisions = []
    all_stack_logprobs = []

    all_sent_logprobs = []
    all_answers = []

    def combine(l1, l2):
        return [x + torch.tensor(y).to(x.device) for x, y in zip(l1, l2)]

    # if silent, then don't print progress bar
    with tqdm(total=len(sents), disable=silent) as progress_bar:
        while st < len(sents):
            en = min(len(sents), st + batch_size)
            cslice = sents[st:en]
            inputs, input_lens = tokenizer_helper(
                lm, tokenizer, data_collator, cslice, parse_slice_or_labels=None
            )
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            # TODO: modify run_greedy_with_stack to return stack_info
            (
                final_preds,
                all_str_logits_curr,
                all_attachment_decisions_curr,
                all_stack_logprobs_curr,
            ) = lm.run_greedy_with_stack(
                inputs,
                input_lens,
                get_str_logits=True,
                get_stack_info=True,
                do_sample=do_sample,
                get_final_answer=get_final_answer,
            )
            if get_final_answer:
                ## bs x max_len x vocab
                preds_curr = [pred.argmax().item() for pred in final_preds]
                all_answers += preds_curr
                all_attachment_decisions += [
                    stack_pred[:l]
                    for stack_pred, l in zip(all_attachment_decisions_curr, input_lens)
                ]
            else:
                logprobs_curr = compute_per_token_logprob(
                    lm, all_str_logits_curr, inputs, input_lens
                )
                all_attachment_decisions += [
                    stack_pred[:l]
                    for stack_pred, l in zip(all_attachment_decisions_curr, input_lens)
                ]
                all_stack_logprobs += all_stack_logprobs_curr
                all_sent_logprobs += combine(logprobs_curr, all_stack_logprobs_curr)

            progress_bar.update(en - st)
            st = en
    if get_final_answer:
        return all_answers, all_attachment_decisions
    else:
        return all_sent_logprobs, all_stack_logprobs, all_attachment_decisions


@torch.no_grad()
def make_preds_base_model(
    lm, tokenizer, sents, gpu_id=0, get_final_answer=False, get_attn_matrices=False
):
    """
    Use language model to make predictions on the given sentences.
    But cannot parse.
    Output:
        - per sentence logprobs
    """

    data_collator = collate.VarLengthCollate(None)
    batch_size = 64
    st = 0
    device = torch.device("cuda:{}".format(gpu_id))
    all_sent_logprobs = []
    all_answers = []
    all_attn_matrices = []
    with tqdm(total=len(sents)) as progress_bar:
        while st < len(sents):
            en = min(len(sents), st + batch_size)
            sent_slice = sents[st:en]
            inputs, input_lens = tokenizer_helper(
                lm, tokenizer, data_collator, sent_slice, parse_slice_or_labels=None
            )
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            if get_attn_matrices:
                outputs = lm.get_attention_matrices(inputs, input_lens)
                all_attn_matrices.append(outputs)
            else:
                outputs = lm(inputs, input_lens)
                all_str_logits_curr = outputs["output"].data
                if get_final_answer:
                    ## bs x max_len x vocab
                    preds_curr = [
                        logit[l - 1].argmax().item()
                        for logit, l in zip(all_str_logits_curr, input_lens)
                    ]
                    all_answers += preds_curr
                else:
                    logprobs_curr = compute_per_token_logprob(
                        lm, all_str_logits_curr.transpose(0, 1), inputs, input_lens
                    )
                    all_sent_logprobs += logprobs_curr
            progress_bar.update(en - st)
            st = en
    if get_final_answer:
        return all_answers
    elif get_attn_matrices:
        return all_attn_matrices
    else:
        return all_sent_logprobs


@torch.no_grad()
def make_preds_with_given_trees(
    lm,
    tokenizer,
    sents,
    parses,
    gpu_id=0,
    silent=True,
    get_final_answer=False,
    stack_type_info=None,
    get_attn_matrices=False,
):
    """
    Use language model to make predictions on the given sentences.
    But, use the given parses.

    Output:
        - perplexity
        - stack_info: logprobs of each of the parses
    """

    data_collator = collate.VarLengthCollate(None)
    batch_size = 32
    st = 0
    device = torch.device("cuda:{}".format(gpu_id))
    all_stack_logprobs = []
    all_sent_logprobs = []
    all_attn_matrices = []
    all_answers = []
    with tqdm(total=len(sents), disable=silent) as progress_bar:
        while st < len(sents):
            en = min(len(sents), st + batch_size)
            sent_slice = sents[st:en]
            parse_slice = parses[st:en]
            import pdb; pdb.set_trace();
            inputs, input_lens, stack_labels, stack_tapes = tokenizer_helper(
                lm,
                tokenizer,
                data_collator,
                sent_slice,
                parse_slice,
                stack_type_info=stack_type_info,
            )
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            stack_tapes = stack_tapes.to(device)
            stack_labels = stack_labels.to(device)

            ### only for synchronous LMs!
            if get_attn_matrices:
                next_words = add_eos(
                    inputs[:, 1:].transpose(0, 1), input_lens - 1, lm.encoder_eos
                ).transpose(0, 1)
                outputs = lm.get_attention_matrices(inputs, input_lens, stack_tapes)
                all_attn_matrices.append(outputs)
            else:
                if lm.synchronous:
                    ### inputs already have sos
                    next_words = add_eos(
                        inputs[:, 1:].transpose(0, 1), input_lens - 1, lm.encoder_eos
                    ).transpose(0, 1)
                    outputs = lm(inputs, next_words, input_lens, stack_tapes)

                else:
                    outputs = lm(inputs, input_lens, stack_tapes)
                all_str_logits_curr = outputs["output"].data

                if get_final_answer:
                    ## bs x max_len x vocab
                    preds_curr = [
                        logit[l - 1].argmax().item()
                        for logit, l in zip(all_str_logits_curr, input_lens)
                    ]
                    all_answers += preds_curr
                else:
                    logprobs_curr = compute_per_token_logprob(
                        lm, all_str_logits_curr.transpose(0, 1), inputs, input_lens
                    )
                    all_stack_logprobs_curr = compute_stack_logprobs(
                        lm, outputs["attachment_logits"][0], stack_labels, input_lens
                    )
                    all_sent_logprobs += logprobs_curr
                    all_stack_logprobs += all_stack_logprobs_curr
            progress_bar.update(en - st)
            st = en
    if get_final_answer:
        return all_answers
    elif get_attn_matrices:
        return all_attn_matrices
    else:
        return all_sent_logprobs, all_stack_logprobs
