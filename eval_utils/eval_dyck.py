from data_utils import build_datasets_dyck
from train_transformers import get_base_transformer_lm
import torch
import hydra
from data_utils.dyck_helpers import read_dyck_data, get_all_prefixes
from omegaconf import DictConfig
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm
from util import set_seed

import beam_search_util


def get_depth_gen_data():
    """Generate dyck data with deep nesting"""
    depth_set = range(15, 50)
    alphabet = "abcdefghij"
    prefs = read_dyck_data(["test"], 20)
    inps = []
    tgts = []
    while True:
        for d in depth_set:
            chosen = []
            inp = []
            for _ in range(d):
                curr = random.choice(alphabet)
                inp += ["({}".format(curr)]
                chosen.append(curr)
            for curr in chosen[::-1]:
                inp += ["{})".format(curr)]
            tgt = inp[-1]
            inp = inp[:-1]
            pref = random.choice(prefs)
            inp.insert(random.randint(0, len(inp)), pref)
            inp = " ".join(inp)
            inps.append(inp)
            tgts.append(tgt)
        if len(inps) >= 100:
            break
    return inps, tgts


def get_long_range_dependencies_data(min_dep_length):
    """Generate dyck data with long range dependencies"""
    sents = read_dyck_data(["train"], 20, max_depth=40)
    prefixes = []
    targets = []
    for sent in sents:
        prefixes_curr, targets_curr = get_all_prefixes(
            sent, min_dep_length=min_dep_length, get_opening_idx=False
        )
        for p, t in zip(prefixes_curr, targets_curr):
            if len(p.split(" ")) > 600:
                continue
            prefixes.append(p)
            targets.append(t)
        if len(prefixes) >= 50:
            break
    return prefixes, targets


@hydra.main(config_path="eval_configs", config_name="dyck_config")
def main(args):
    set_seed(42)
    _, in_vocab, _ = build_datasets_dyck(do_compute_stack_labels=False)
    lm, _ = get_base_transformer_lm(
        args,
        in_vocab,
        model_name=args.model_name,
        recursive_layer_args=args.recursive_layer,
    )

    device = torch.device("cuda:{}".format(args.gpu_id))
    lm.to(device)
    lm.eval()

    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    if args.eval_type == "long_range":
        print(
            "Performing long range dependency evaluation with min length {}".format(
                args.get("dep_length", 50)
            )
        )
        prefixes, targets = get_long_range_dependencies_data(args.get("dep_length", 50))
    else:
        print("Performing depth generalization evaluation")
        prefixes, targets = get_depth_gen_data()
    beam_obj = beam_search_util.BeamSearch(50, input_is_prefix=True)
    vocab_items_closing_brackets = [
        in_vocab.words[s] for s in in_vocab.words if ")" in s
    ]
    accs = 0
    for sent, target in tqdm(zip(prefixes, targets), total=len(prefixes)):
        out, preds = beam_obj(lm, tokenizer, sent, 0)
        preds_closing = vocab_items_closing_brackets[
            preds[0][vocab_items_closing_brackets].argmax()
        ]
        accs += preds_closing == in_vocab(target)[0]

    print(accs / len(prefixes))


if __name__ == "__main__":
    main()
