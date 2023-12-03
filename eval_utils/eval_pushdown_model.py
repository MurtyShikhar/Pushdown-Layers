# evaluate perplexity of a pushdown model, using either beam search or gold parses

import sys
from pathlib import Path

# This line adds the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from transformers import GPT2Tokenizer
import hydra
import torch
import numpy as np
import argparse
import yaml
import random, pickle
import pushdown_util

import os
from data_utils import build_datasets_pushdown
from data_utils.text_helpers import binarize_tree, flatten
from omegaconf import DictConfig
from transformer_helpers import create_lm
from munch import Munch
from util import set_seed, convert_tree_to_tuple
from tqdm import tqdm
from util import get_base_transformer_lm
from data_utils.text_helpers import attachment_decisions_to_tree

from nltk import Tree
import beam_search_util


class PreProcessor:
    def __init__(self):
        # tokenize with GPT2 tokenizer
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def process(self, ex, add_eos=True):
        # if last word has question mark, remove it

        if ex[-1] == "?":
            ex = ex[:-1] + " ?"
        tokenized = self.gpt_tokenizer.tokenize(ex)
        if add_eos:
            joined_ex = " ".join(tokenized) + " <eos>"
        else:
            joined_ex = " ".join(tokenized)
        return joined_ex


def get_parsing_accuracy(predicted_parses, gold_parses):
    """Compute parsing scores for predicted parses."""

    def get_brackets(parse):
        p_set = set()

        def get_brackets_helpers(t, st):
            if type(t) == str:
                return 1
            else:
                l1_len = get_brackets_helpers(t[0], st)
                l2_len = get_brackets_helpers(t[1], st + l1_len)
                p_set.add((st, st + l1_len + l2_len - 1))
                return l1_len + l2_len

        get_brackets_helpers(parse, 0)
        return p_set

    gold_brackets = [get_brackets(parse) for parse in gold_parses]
    pred_brackets = [get_brackets(parse) for parse in predicted_parses]

    def get_score(set_1, set_2):
        score = 0.0
        for p in set_2:
            if p in set_1:
                score += 1
        return score

    precision = sum(
        [get_score(gold, pred) for gold, pred in zip(gold_brackets, pred_brackets)]
    )
    recall = sum(
        [get_score(pred, gold) for gold, pred in zip(gold_brackets, pred_brackets)]
    )
    precision /= 1.0 * sum(len(b) for b in pred_brackets)
    recall /= 1.0 * sum(len(b) for b in gold_brackets)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall + 1e-10),
    }


def load_data(args):
    if "bllip" in args.dataset:
        with open("../data_utils/{}/{}.txt".format(args.dataset, args.data_split)) as f:
            data = [Tree.fromstring(l.strip()) for l in f.readlines()]
        examples = [(d, flatten(d, add_eos=True)) for d in data]
        return examples
    elif "blimp" in args.dataset:
        # this is blimp data, where each example is a tuple of (goodsent, badsent, phenomena)
        with open("../data_utils/blimp/benepar_parses_depth_or_typed.pkl", "rb") as f:
            data = [Tree.fromstring(t) for t in pickle.load(f)]
        examples = [(convert_tree_to_tuple(d), flatten(d, add_eos=True)) for d in data]
        return examples
    else:
        preprocessor = PreProcessor()
        with open(args.dataset, "r") as reader:
            data = [preprocessor.process(l.strip()) for l in reader.readlines()]
        return data


def callback_pushdown(lm, in_vocab, split, with_pushdown=True, data_folder_given=None):
    """Callback function for pushdown lm training."""
    DATA_DIR = "/u/scr/smurty/pushdown-lm"
    if data_folder_given:
        folder_dir = data_folder_given
    else:
        folder_dir = "bllip-lg-depth"
    with open("{}/data_utils/{}/{}.txt".format(DATA_DIR, folder_dir, split)) as f:
        data = [Tree.fromstring(l.strip()) for l in f.readlines()]

    # create tokenizer
    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    device = torch.device("cuda:0")
    examples = [(d, flatten(d, add_eos=True)) for d in data]

    # the callback function returns ppl with ground truth parses
    if with_pushdown:
        sent_ppl_gold, _ = eval_gold(
            lm,
            examples,
            tokenizer,
        )
        return {
            "gold_ppl": sent_ppl_gold,
        }
    else:
        sent_ppl, _ = eval_base_model(lm, examples, tokenizer, device)
        return {"ppl": sent_ppl}


def compute_perplexity_from_logprobs(all_logprobs):
    """
    Compute perplexity from logprobs. works for both torch and numpy arrays.
    Also works if we want to marginalize parses
    """
    if type(all_logprobs[0]) == torch.Tensor:
        total_logprob = np.sum([torch.sum(p).item() for p in all_logprobs])
        total_len = np.sum([len(s) for s in all_logprobs])
    elif len(all_logprobs[0]) == 2:
        ### sent logprb, length
        total_len = np.sum([_len for logprob_set, _len in all_logprobs])
        total_logprob = np.sum(
            [logsumexp(logprob_set) for logprob_set, _len in all_logprobs]
        )
    else:
        total_logprob = np.sum([np.sum(p) for p in all_logprobs])
        total_len = np.sum([len(s) for s in all_logprobs])
    return np.exp(-total_logprob / total_len)


def eval_base_model(lm, examples, tokenizer, device):
    """Evaluate a standard transformer LM (no pushdown / external stack)."""
    all_sent_logprobs = pushdown_util.make_preds_base_model(
        lm, tokenizer, [s for p, s in examples]
    )
    sent_ppl = compute_perplexity_from_logprobs([x[:-1] for x in all_sent_logprobs])
    return sent_ppl, None


def eval_gold(
    lm,
    examples,
    tokenizer,
):
    all_sent_logprobs, all_stack_logprobs = pushdown_util.make_preds_with_given_trees(
        lm,
        tokenizer,
        [s for _, s in examples],
        [p for p, _ in examples],
        stack_type_info="depth",
    )

    sent_logprobs = [
        (np.sum(logprob) + np.sum(stack_logprob), len(stack_logprob))
        for logprob, stack_logprob in zip(all_sent_logprobs, all_stack_logprobs)
    ]

    total_len = np.sum([l for _, l in sent_logprobs])
    sent_ppl = np.exp(-np.sum([logprob for logprob, _ in sent_logprobs]) / total_len)
    stack_ppl = compute_perplexity_from_logprobs(all_stack_logprobs)
    return sent_ppl, stack_ppl


def logsumexp(x):
    x = np.array(x)
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))


def eval_beam(
    lm,
    examples,
    tokenizer,
    device,
    write_dir=None,
    marginalize=False,
    beam_sz=300,
):
    print("Using depth version of beam search")
    beam_search = beam_search_util.BeamSearchDepthBased(beam_sz)

    def beam_search_helper(sent):
        final_beam = beam_search(lm, tokenizer, sent, 0)
        if marginalize:
            # marginalize over all possible parses
            all_joint_logprobs = []
            attachment_decisions = [0] + final_beam[0][1]
            all_attachment_logprobs = []
            for beam in final_beam:
                # this is the joint logprob of sent, parse i.e. P(sent, parse)
                sent_parse_logprobs = beam[0]
                attachment_logprobs = np.sum(
                    [attachment_logprob for _, attachment_logprob in beam[2]]
                )
                all_joint_logprobs.append(sent_parse_logprobs)
                all_attachment_logprobs.append(attachment_logprobs)

            # The one is subtracted because we ignore the <eos> => </s> transition (superfluous)
            return (
                [all_joint_logprobs, len(attachment_decisions) - 1],
                all_attachment_logprobs,
                attachment_decisions,
            )
        else:
            best_beam = final_beam[0]
            # joint logprob of sentence and stack!
            sent_parse_logprobs = [sent_logprob for sent_logprob in best_beam[2]]
            attachment_logprobs = [lp for _, lp in best_beam[2]]
            attachment_decisions = [0] + best_beam[1]
            return sent_parse_logprobs, attachment_logprobs, attachment_decisions

    print("running beam search")
    all_joint_logprobs = []
    all_attachment_decisions = []
    all_attachment_logprobs = []
    for example in tqdm(examples):
        if type(example) == tuple:
            parse, sent = example
        else:
            parse = None
            sent = example

        (
            all_joint_logprobs_curr,
            attachment_logprobs_curr,
            all_attachment_decisions_curr,
        ) = beam_search_helper(sent)

        all_joint_logprobs.append(all_joint_logprobs_curr)
        all_attachment_logprobs.append(attachment_logprobs_curr)
        all_attachment_decisions.append(all_attachment_decisions_curr)

        if len(all_joint_logprobs) % 100 == 0 and write_dir is None and parse is None:
            sent_ppl = compute_perplexity_from_logprobs(all_joint_logprobs)
            if isinstance(examples[0][0], Tree):
                gold_parses = [
                    convert_tree_to_tuple(p)
                    for p, s in examples[: len(all_joint_logprobs)]
                ]
            else:
                gold_parses = [p for p, s in examples[: len(all_joint_logprobs)]]
            predicted_parses = [
                attachment_decisions_to_tree(pred, ex[1])
                for pred, ex in zip(all_attachment_decisions, examples)
            ]
            parsing_acc = get_parsing_accuracy(predicted_parses, gold_parses)
            print("curr_sent_ppl", sent_ppl)
            print("parsing acc", parsing_acc["f1"])

    if type(examples[0]) == tuple:
        predicted_parses = [
            attachment_decisions_to_tree(pred, ex[1])
            for pred, ex in zip(all_attachment_decisions, examples)
        ]
    else:
        predicted_parses = [
            attachment_decisions_to_tree(pred, ex)
            for pred, ex in zip(all_attachment_decisions, examples)
        ]

    if write_dir is not None:
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        with open(os.path.join(write_dir, "beam_search.pkl"), "wb") as f:
            pickle.dump(
                {
                    "all_sent_logprobs": all_joint_logprobs,
                    "all_attachment_decisions": all_attachment_decisions,
                    "pred_parses": predicted_parses,
                    "all_attachment_logprobs": all_attachment_logprobs,
                },
                f,
            )
        # write trees to "trees.txt"
        with open(os.path.join(write_dir, "trees.txt"), "w") as f:
            for pred in predicted_parses:
                f.write(str(pred) + "\n")

    sent_ppl = compute_perplexity_from_logprobs(all_joint_logprobs)
    if isinstance(examples[0], tuple):
        if isinstance(examples[0][0], Tree):
            gold_parses = [convert_tree_to_tuple(p) for p, s in examples]
        else:
            gold_parses = [p for p, s in examples]

        parsing_acc = get_parsing_accuracy(predicted_parses, gold_parses)
    else:
        parsing_acc = None
    # no stack_ppl for beam search...
    return sent_ppl, 0.0, parsing_acc


@hydra.main(config_path="eval_configs", config_name="bllip_config")
def main(all_args):
    args = all_args.settings
    recursive_layer_args = all_args.recursive_layer
    all_args["train"]["model_load_path"] = "{}/model.pt".format(args.model_dir)
    set_seed(42)

    # load model
    in_vocab = pickle.load(open("{}/vocab.pkl".format(args.model_dir), "rb"))
    lm, interface = get_base_transformer_lm(
        all_args.train,
        in_vocab,
        model_name=all_args.train.model_load_path,
        recursive_layer_args=recursive_layer_args,
    )

    device = torch.device("cuda:{}".format(all_args.train.gpu_id))
    lm.to(device)
    lm.eval()

    # create tokenizer
    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    # load data
    examples = load_data(args)

    if args.eval_base:
        sent_ppl, stack_ppl = eval_base_model(lm, examples, tokenizer, device)
    elif args.eval_mode == "gold":
        sent_ppl, stack_ppl = eval_gold(lm, examples, tokenizer)
    elif args.eval_mode == "beam":
        if args.shard_id != -1:
            # we use sharding because beam search can be slow.
            # use args.shard_id and args.num_shards to find the shard of the data to use
            slice_size = len(examples) // args.num_shards
            slice_st = args.shard_id * slice_size
            slice_en = (
                (args.shard_id + 1) * slice_size
                if args.shard_id != args.num_shards - 1
                else len(examples)
            )
            examples = examples[slice_st:slice_en]
            sent_ppl, stack_ppl, parsing_acc = eval_beam(
                lm,
                examples,
                tokenizer,
                device,
                args.dir_name,
                marginalize=args.marginalize,
                beam_sz=args.beam_size,
            )
        else:
            sent_ppl, stack_ppl, parsing_acc = eval_beam(
                lm,
                examples,
                tokenizer,
                device,
                write_dir=args.dir_name if args.dir_name != "" else None,
                marginalize=args.marginalize,
            )
        print(parsing_acc)
    else:
        raise ValueError("Method not implemented.")

    print("sentence ppl: ", sent_ppl)
    print("stack ppl: ", stack_ppl)


if __name__ == "__main__":
    main()
