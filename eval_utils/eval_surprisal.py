from transformers import GPT2Tokenizer
import numpy as np
import re
import json
from tqdm import tqdm
import argparse
import math
import models
from munch import Munch
import hydra

from util import get_base_transformer_lm
from data_utils.text_helpers import (
    attachment_decisions_to_tree,
)

import torch
import os
import pickle
from beam_search_util import BeamSearchDepthBased, logsumexp

from pushdown_util import (
    make_preds_base_model,
    make_preds_with_given_trees,
)
from nltk import Tree


class PreProcessor:
    def __init__(self):
        ### tokenize with GPT2 tokenizer
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def process(self, ex, add_eos=True):
        ### if last word has question mark, remove it

        if ex[-1] == "?":
            ex = ex[:-1] + " ?"
        tokenized = self.gpt_tokenizer.tokenize(ex)
        if add_eos:
            joined_ex = " ".join(tokenized) + " <eos>"
        else:
            joined_ex = " ".join(tokenized)
        return joined_ex


def eval_math_expr(expr):
    try:
        return eval(expr)
    except:
        return math.nan


def dummy_tree(parse):
    def helper(p):
        if type(p) == str:
            return "(X {})".format(p) if p not in ["(", ")"] else "(X paren)"
        else:
            out = " ".join(helper(x) for x in p)
            return "(X {})".format(out)

    return helper(parse)


class TestSuiteParser:
    def __init__(self, test_suite_file):
        self.test_suite_file = test_suite_file
        self.read_test_suite()
        self.answers = [0 for _ in range(len(self.meta_data["data"]))]

    def read_test_suite(self):
        data_file = (
            "/u/scr/smurty/syntactic-generalization/test_suites/json/{}.json".format(
                self.test_suite_file
            )
        )
        with open(data_file, "r") as f:
            data = json.load(f)
        self.meta_data = {
            "formula": data["predictions"][0]["formula"],
            "data": self.get_sents(data),
        }

    def get_sents(self, data):
        all_ex = []
        for item in data["items"]:
            curr_ex = {}
            for cond in item["conditions"]:
                regions = [x["content"] for x in cond["regions"]]
                curr_ex[cond["condition_name"]] = regions
            all_ex.append(curr_ex)
        return all_ex

    def extract_formulas(self, surprisal_dict):
        formula = self.meta_data["formula"]
        keys = re.findall(r"%([\w|-]+)%", formula)
        keys = set(keys)
        for key in keys:
            positions = set(re.findall(r"\((\d+);%{}%".format(key), formula))
            for position in positions:
                formula = formula.replace(
                    "({};%{}%)".format(position, key),
                    str(surprisal_dict[key][int(position)]),
                )
        ### replace [ with ( and ] with ) to make it a valid math expression

        formula = formula.replace("[", "(")
        formula = formula.replace("]", ")")
        return formula

    def get_example(self, idx):
        return self.meta_data["data"][idx]

    def evaluate_example(self, idx, evaluator, verbose=False):
        examples = self.get_example(idx)
        phen2surprisals = {}
        for phen in examples:
            target_surprisals, logprobs, target_idxs, _ = evaluator.get_surprisals(
                examples[phen]
            )
            if verbose:
                print("Regions: {}".format(examples[phen]))
                print(logprobs)
            phen2surprisals[phen] = [0] + target_surprisals

        extracted_formula = self.extract_formulas(phen2surprisals)
        self.answers[idx] = extracted_formula

    def evaluate_all(self, evaluator):
        for idx in tqdm(range(len(self.meta_data["data"]))):
            self.evaluate_example(idx, evaluator)
        return


class Evaluator:
    def __init__(
        self,
        lm,
        beam_obj,
        tokenizer,
        non_incremental=False,
        benepar_obj=None,
        stack_type_info=None,
    ):
        ### lm: language model, beam_obj: beam search object, tokenizer: tokenizer corresponding to LM
        self.lm = lm
        self.beam_obj = beam_obj
        self.preprocessor = PreProcessor()
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer = tokenizer
        self.benepar_obj = benepar_obj
        self.stack_type_info = stack_type_info
        self.non_incremental = non_incremental

    def run_beam_search(self, sent_processed, get_surprisal=True):
        if get_surprisal:
            logprobs, best_incremental_parses, beams = self.beam_obj(
                self.lm, self.tokenizer, sent_processed, 0, get_surprisal=get_surprisal
            )
            return logprobs, best_incremental_parses, beams
        else:
            beams = self.beam_obj(
                self.lm, self.tokenizer, sent_processed, 0, get_surprisal=get_surprisal
            )
            return beams

    def marginalize(self, beams):
        num_words = len(beams[0][2])

        partial_logprob = [0.0]
        prev_marginalized_logprob = 0.0
        marginalized_logprobs = []
        for i in range(num_words):
            curr_logprob_set = []
            seen_parses = set()
            for beam in beams:
                partial_parse = tuple(beam[1][: i + 1])
                if partial_parse not in seen_parses:
                    seen_parses.add(partial_parse)
                    curr_logprob = np.sum(beam[2][:i])
                    if beam[1][i] != i:
                        curr_logprob += beam[2][i][0]
                    else:
                        # shift action!
                        curr_logprob += beam[2][i][0] + beam[2][i][1]
                    curr_logprob_set.append(curr_logprob)
            curr_marginalized_logprob = logsumexp(curr_logprob_set)
            marginalized_logprobs.append(
                curr_marginalized_logprob - prev_marginalized_logprob
            )
            prev_marginalized_logprob = curr_marginalized_logprob
        return marginalized_logprobs

    def get_target_idxs(self, regions):
        sent = " ".join([r.lstrip().rstrip() for r in regions if len(r) > 0])

        sent_processed = self.preprocessor.process(sent + " .")
        region_subword_lens = [
            len(
                self.gpt_tokenizer.tokenize(
                    x.lstrip().rstrip(),
                    add_prefix_space=idx != 0 and len(x.lstrip().rstrip()) > 0,
                )
            )
            for idx, x in enumerate(regions)
        ]
        cumulative_region_subword_lens = [0] + list(np.cumsum(region_subword_lens))
        all_target_idxs = []
        for idx, region in enumerate(regions):
            t_start = cumulative_region_subword_lens[idx]
            t_end = cumulative_region_subword_lens[idx + 1]
            all_target_idxs.append((t_start, t_end))
        return all_target_idxs

    def get_surprisals(self, regions, verbose=False):
        ### a list of regions which when concatenated with a period and processed by the preprocessor, gives a valid input to the language model
        ### but some regions can be empty, so we need to take care of that

        sent = " ".join([r.lstrip().rstrip() for r in regions if len(r) > 0])

        sent_processed = self.preprocessor.process(sent + " .")
        region_subword_lens = [
            len(
                self.gpt_tokenizer.tokenize(
                    x.lstrip().rstrip(),
                    add_prefix_space=idx != 0 and len(x.lstrip().rstrip()) > 0,
                )
            )
            for idx, x in enumerate(regions)
        ]
        cumulative_region_subword_lens = [0] + list(np.cumsum(region_subword_lens))
        all_target_idxs = []
        for idx, region in enumerate(regions):
            t_start = cumulative_region_subword_lens[idx]
            t_end = cumulative_region_subword_lens[idx + 1]
            all_target_idxs.append((t_start, t_end))

        if self.benepar_obj:
            parse = self.benepar_obj(sent + " .")
            try:
                sent_logprobs, parse_logprobs = make_preds_with_given_trees(
                    self.lm,
                    self.tokenizer,
                    [sent_processed],
                    [parse],
                    stack_type_info=self.stack_type_info,
                )
            except:
                print(parse)
                print(regions)
                print(sent_processed)
                parse.pretty_print()
                print(sent)

            sent_logprobs = sent_logprobs[0]
            parse_logprobs_prev = [0] + parse_logprobs[0][:-1]
            logprobs = [s + p for s, p in zip(sent_logprobs, parse_logprobs_prev)]
            best_incremental_parses = None
        elif self.beam_obj:
            if not self.non_incremental:
                logprobs, best_incremental_parses, _ = self.run_beam_search(
                    sent_processed
                )
            else:
                beams = self.run_beam_search(sent_processed, get_surprisal=False)
                # marginalize over all beams now.
                logprobs = self.marginalize(beams)
                best_incremental_parses = None
        else:
            all_sent_logprobs = make_preds_base_model(
                self.lm, self.tokenizer, [sent_processed]
            )
            logprobs = all_sent_logprobs[0]
        target_surprisals = [
            -1.0 * np.sum(logprobs[st:en]) for st, en in all_target_idxs
        ]

        if verbose:
            words = sent_processed.split()
            ### pretty print word and corresponding logprob to 2 decimal places
            print(["{}: {:.2f}".format(w, l) for w, l in zip(words, logprobs)])

        return target_surprisals, logprobs, all_target_idxs, best_incremental_parses

    def get_sent_logprob(self, regions, verbose=False):
        sent = " ".join([r.lstrip() for r in regions if len(r) > 0])
        sent_processed = self.preprocessor.process(sent + " .")
        beams = self.run_beam_search(sent_processed, get_surprisal=False)

        total_logprob = [b[0] for b in beams]

        if verbose:
            print(sent_processed)
            for b in beams[:3]:
                best_parse = attachment_decisions_to_tree(b[1], sent_processed)
                tree = Tree.fromstring(dummy_tree(best_parse))
                tree.pretty_print()

        return logsumexp(total_logprob), beams


@hydra.main(config_path="eval_configs", config_name="bllip_config")
def main(all_args):
    args = all_args.settings
    recursive_layer_args = all_args.recursive_layer
    all_args["train"]["model_load_path"] = "{}/model.pt".format(args.model_dir)
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

    ### create tokenizer
    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    if args.eval_base:
        beam_obj = None
    else:
        beam_obj = BeamSearchDepthBased(300)

    eval_obj = Evaluator(lm, beam_obj, tokenizer, non_incremental=args.non_incremental)

    test_suite_parser = TestSuiteParser(args.test_suite_name)
    test_suite_parser.evaluate_all(eval_obj)

    acc = 0.0
    for formula in test_suite_parser.answers:
        acc += eval_math_expr(formula)

    print(acc / len(test_suite_parser.answers))
    if args.dir_name != "":
        result = {
            "answers": test_suite_parser.answers,
            "acc": acc / len(test_suite_parser.answers),
        }
        with open(os.path.join(args.dir_name, "answers.pkl"), "wb") as f:
            pickle.dump(result, f)


if __name__ == "__main__":
    main()
