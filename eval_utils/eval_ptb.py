# Quick (and dirty) script to load a model and generate parses for PTB
# With sharding, so we can run multiple evals in parallel
# Usage: cd eval_utils.py; python eval_ptb.py --shard_id 0 --num_shards 1 --dir_name ptb_results
# The script will write a pickle file containing model predictions and
# ground truth parses (processed such that words are GPT2-tokenized and everything is CNFed)

import sys
from pathlib import Path

# This line adds the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from munch import Munch
import hydra
import yaml
import os, pickle
from util import get_base_transformer_lm
import argparse
from tqdm import tqdm
from transformers import GPT2Tokenizer
import torch
from nltk import Tree
from eval_surprisal import Evaluator, TestSuiteParser
from beam_search_util import BeamSearchDepthBased
from data_utils.text_helpers import attachment_decisions_to_tree


# util to convert nltk tree into nested tuples
def convert_into_tuple(nltk_tree):
    """
    Convert NLTK tree into a tuple of tuples
    """

    def helper(tree):
        if type(tree) == str:
            return tree
        elif len(tree) == 1:
            return helper(tree[0])
        else:
            return tuple(helper(x) for x in tree)

    return helper(nltk_tree)


# Load test set
class ParserPipeline:
    def __init__(self):
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def post_process_op(self, tree):
        """
        Input: nltk tree where leaf nodes are strings
        Output: nltk tree but use gpt_tokenizer to tokenize leaf nodes and create a subtree corresponding to it.
        """

        def fix(t, is_first, label=None):
            if type(t) == str:
                tokenized = self.gpt_tokenizer.tokenize(
                    t, add_prefix_space=not is_first
                )
                if len(tokenized) == 1:
                    return Tree(label, [tokenized[0]])
                else:
                    return Tree(label, [Tree(label, [tok]) for tok in tokenized])
            elif len(t) == 1:
                return fix(t[0], is_first=is_first, label=t.label())
            else:
                return Tree(
                    t.label(),
                    [fix(c, is_first=is_first and idx == 0) for idx, c in enumerate(t)],
                )

        fixed_tree = fix(tree, is_first=True)
        fixed_tree.chomsky_normal_form()
        return fixed_tree

    def process(self, parse):
        ptree = Tree.fromstring(parse)
        ptree.chomsky_normal_form()
        return self.post_process_op(ptree)

    def __call__(self, parse):
        return self.process(parse)


@hydra.main(config_path="eval_configs", config_name="bllip_config")
def main(all_args):
    args = all_args.settings
    shard_id = args.shard_id
    num_shards = args.num_shards
    # Load model
    in_vocab = pickle.load(open("{}/vocab.pkl".format(args.model_dir), "rb"))
    lm, interface = get_base_transformer_lm(
        all_args.train,
        in_vocab,
        model_name=all_args.train.model_load_path,
        recursive_layer_args=all_args.recursive_layer,
    )

    device = torch.device("cuda:{}".format(all_args.train.gpu_id))
    lm.to(device)
    lm.eval()

    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    pipeline = ParserPipeline()
    PTB_DATA_PATH = "ptb_samples_small.mrg"
    with open(PTB_DATA_PATH, "r") as reader:
        data = [pipeline(l.strip()) for l in reader.readlines()]
    beam_obj = BeamSearchDepthBased(args.beam_size)
    ### start and end for this shard
    shard_id = args.shard_id
    num_shards = args.num_shards
    data_slice_st = len(data) * shard_id // num_shards
    data_slice_en = len(data) * (shard_id + 1) // num_shards
    print("st: {}. en: {}".format(data_slice_st, data_slice_en))
    data = data[data_slice_st:data_slice_en]
    # evaluate
    ground_truth_parses = []
    our_parses = []
    for ex in tqdm(data):
        dat = " ".join(ex.leaves()) + " <eos>"
        try:
            final_beam = beam_obj(lm, tokenizer, dat, 0)
            parse = attachment_decisions_to_tree([0] + final_beam[0][1], dat)
            ground_truth_parses.append(convert_into_tuple(ex))
            our_parses.append(parse)
        except Exception as e:
            print(e)
            continue

    with open("{}/{}.pickle".format(args.dir_name, shard_id), "wb") as writer:
        pickle.dump((ground_truth_parses, our_parses), writer)


if __name__ == "__main__":
    main()
