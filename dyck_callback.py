# Bunch of callbacks for evaluating pushdown LMs when trained on Dyck Languages
from data_utils.dyck_helpers import read_dyck_data, get_all_prefixes, compute_stack_labels, compute_stack_tape
from tqdm import tqdm
import torch
import collate
import torch.nn.functional as F


# On this data, the tokenizer just adds an <SOS> token to the front of the string
def tokenizer_helper(lm, tokenizer,
                     data_collator,
                     inp_slice_and_target,
                     pred_stack_reprs=None):
    inp_slice, full_target = inp_slice_and_target
    inp_list = [tokenizer(s) for s in inp_slice]
    out_list = [tokenizer(s, add_sos=False) for s in full_target]

    in_lens = [len(s) for s in inp_list]

    stack_labels = [compute_stack_labels(s)[1][1:] for s in inp_slice]
    if lm.trafo.use_stack_tape:
        if pred_stack_reprs is not None:
            stack_reprs = pred_stack_reprs
        else:
            stack_reprs = [
                compute_stack_tape(s) for s in stack_labels
            ]
        inp_to_collate = [
            {"in": x, "target": y, "in_len": in_len, "stack_repr": stack, "stack_label": stack_label}
            for x, y, in_len, stack, stack_label in zip(
                inp_list, out_list, in_lens, stack_reprs, stack_labels
            )
        ]
    else:
        inp_to_collate = [
            {"in": x, "target": y, "in_len": in_len, "stack_label": stack_label}
            for x, y, in_len, stack_label in zip(
                inp_list, out_list, in_lens, stack_labels
            )
        ]

    inp = data_collator(inp_to_collate)
    in_len = inp["in_len"].long()
    if lm.trafo.use_stack_tape:
        return (
            inp["in"].transpose(0, 1),
            inp["target"].transpose(0, 1),
            in_len,
            inp["stack_label"].transpose(0, 1),
            inp["stack_repr"].transpose(0, 1),
        )
    else:
        return (
            inp["in"].transpose(0, 1),
            inp["target"].transpose(0, 1),
            in_len,
            inp["stack_label"].transpose(0, 1),
            None,
        )

@torch.no_grad()
def test_continuations(
    tokenizer,
    lm,
    prefixes,
    full_target,
    gpu_id,
    attn_layer=-1,
    pred_syntax=None,
):
    # e.g.:
    # a prefix is (a (b (c c), target is b)
    # after tokenization:
    # <s> (a (b (c c)
    # Full-target: (a (b (c c) b)

    data_collator = collate.VarLengthCollate(None)

    batch_size = 8
    st = 0
    device = torch.device("cuda:{}".format(gpu_id))

    final_states = []

    attachment_decision_correct = 0.0
    attachment_decision_total = 0.0

    with tqdm(total=len(prefixes)) as progress_bar:
        while st < len(prefixes):
            en = min(len(prefixes), st + batch_size)
            cslice = prefixes[st:en], full_target[st:en]

            if pred_syntax is not None:
                inputs, targets, input_lens, stack_labels, stack_tape = tokenizer_helper(
                    lm, tokenizer, data_collator, cslice, pred_syntax[st:en]
                )
            else:
                inputs, targets, input_lens, stack_labels, stack_tape = tokenizer_helper(
                    lm, tokenizer, data_collator, cslice
                )
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            targets = targets.to(device)
            if lm.trafo.use_stack_tape:
                stack_tape = stack_tape.to(device)
            else:
                stack_tape = None
            output_dict = lm(inputs, targets, input_lens, stack_tape)
            outputs = output_dict["output"]
            attachment_logits = output_dict["attachment_logits"][0]
            attachment_decisions = attachment_logits.argmax(axis=-1)
            # add a 0 to make the shape bs x seq_len+1
            stack_labels = torch.cat(
                [
                    stack_labels,
                    torch.zeros(
                        (stack_labels.shape[0], 1), dtype=stack_labels.dtype
                    ),
                ],
                dim=1,
            ).to(attachment_decisions.device)
            attachment_decision_correct += (
                ((attachment_decisions == stack_labels) * (stack_labels != 0))
                .sum()
                .item()
            )
            attachment_decision_total += (stack_labels != 0).sum().item()


            final_states += [
                outputs["data"][idx][l - 1] for idx, l in enumerate(input_lens)
            ]
            progress_bar.update(en - st)
            st = en
    final_states = torch.stack(final_states, dim=0)
    return F.softmax(final_states, dim=1), attachment_decision_correct / attachment_decision_total

def test_attachment_decisions(
    tokenizer,
    lm,
    sents,
    gpu_id,
    get_preds=False,
    batch_size=8,
    pred_syntax=None,
    is_dyck=True,
):
    data_collator = collate.VarLengthCollate(None)
    st = 0
    device = torch.device("cuda:{}".format(gpu_id))
    # the tokenizer just adds an <SOS> token to the front of the string
    correct = 0
    total = 0

    attachment_decisions_all = []
    with tqdm(total=len(sents), disable=get_preds) as progress_bar:
        while st < len(sents):
            en = min(len(sents), st + batch_size)
            cslice = sents[st:en]
            if pred_syntax is not None:
                inputs, input_lens, stack_labels, stack_repr = tokenizer_helper(
                    lm,
                    tokenizer,
                    data_collator,
                    cslice,
                    pred_syntax[st:en],
                )
            else:
                inputs, input_lens, stack_labels, stack_repr = tokenizer_helper(
                    lm, tokenizer, data_collator, cslice)
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            if lm.trafo.use_stack_tape:
                stack_repr = stack_repr.to(device)
            else:
                stack_repr = None

            stack_labels = stack_labels.to(device)
            with torch.no_grad():
                outputs = lm(inputs, input_lens, stack_repr)["stack_logits"][
                    0
                ]  # .transpose(1, 2)
                if get_preds:
                    # transpose back
                    assert batch_size == 1
                    attachment_decisions_all.append(outputs[0])  # .transpose(0, 1))
                else:
                    attachment_decisions = outputs.argmax(axis=-1)
                    correct += (
                        ((attachment_decisions == stack_labels) * (stack_labels != 0))
                        .sum()
                        .item()
                    )
                    total += (stack_labels != 0).sum().item()
            progress_bar.update(en - st)
            st = en

    if total:
        acc = correct / total
    else:
        acc = 0.0
    if get_preds:
        return {"attachment_decisions": attachment_decisions_all, "stack_acc": acc}
    else:
        return {"stack_acc": acc}

def eval_callback_dyck(lm, in_vocab, split, eval_attachment_decisions=False):
    def tokenizer(s, add_sos=True):
        if add_sos:
            return [lm.encoder_sos] + in_vocab(s)
        else:
            return in_vocab(s)

    prefixes = []
    targets = []

    if split == "test":
        sents = read_dyck_data(["train"], 20, max_depth=40)
    else:
        sents = read_dyck_data([split], 20)
    for sent in sents:
        if split == "val":
            min_dep_length = None
        else:
            min_dep_length = 50
        prefixes_curr, targets_curr = get_all_prefixes(
            sent, min_dep_length=min_dep_length, get_opening_idx=False
        )
        for p, t in zip(prefixes_curr, targets_curr):
            if len(p.split(" ")) > 250:
                continue
            prefixes.append(p)
            targets.append(t)
        if len(prefixes) >= 500:
            break

    full_target = ['{} {}'.format(prefix, target) for prefix, target in zip(prefixes, targets)]
    out, stack_pred_accuracy = test_continuations(tokenizer, lm, prefixes, full_target, 0)
    vocab_items_closing_brackets = [
        in_vocab.words[s] for s in in_vocab.words if ")" in s
    ]

    out_closing = out[:, vocab_items_closing_brackets]
    best_closing_entry = [
        vocab_items_closing_brackets[idx] for idx in out_closing.argmax(dim=1)
    ]
    accs = [pred == in_vocab(t)[0] for pred, t in zip(best_closing_entry, targets)]
    agg_acc = sum(accs) / len(prefixes)

    stack_pred_acc = {'stack_acc': stack_pred_accuracy}

    acc_dict = {"acc": agg_acc}
    return {**acc_dict, **stack_pred_acc}
