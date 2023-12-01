import numpy as np
import collate
import random
import pickle
import os
import torch
from training_utils import *

import argparse
from data_utils import (
    build_datasets_dyck,
    build_datasets_pushdown,
)
from transformer_helpers import *
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig

from eval_utils.eval_pushdown_model import callback_pushdown
from dyck_callback import eval_callback_dyck
from util import get_base_transformer_lm


### Change this for your own system as appropriate
def working_dir():
    USER = os.environ["USER"]
    dir_name = f"/scr/biggest"

    def helper(dir_name):
        if os.path.exists(dir_name):
            sub_dir = "{}/{}/compositionality".format(dir_name, USER)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            return sub_dir
        else:
            return ""

    try:
        return helper(dir_name)
    except:
        dir_name = f"/scr"
        return helper(dir_name)


@hydra.main(config_path="config", config_name="config")
def main_lm(all_args):
    args = all_args.train
    set_seed(args)
    wandb.run.name = "{}-{}".format(args.save_dir, args.seed)
    wandb.run.save()
    recursive_layer_args = all_args.recursive_layer

    if args.dataset == "dyck":
        datasets, in_vocab, _ = build_datasets_dyck(
            vocab=args.dyck_vocab,
            use_stack_tape=args.use_stack_tape,
            data_regime=args.get("data_regime", "normal"),
        )
        stack_type_vocab = None
    else:
        ### if using the parses in any way
        (
            datasets,
            in_vocab,
            _,
        ) = build_datasets_pushdown(
            data_regime=args.get("data_regime", "normal"),
            do_compute_attachment_labels=recursive_layer_args.get(
                "attachment_decisions", False
            ),
            use_stack_tape=args.use_stack_tape,
            data_file_given=args.dataset,
            with_depth_info=args.get("with_depth_info", False),
        )
    model, interface = get_base_transformer_lm(
        args,
        in_vocab,
        recursive_layer_args=recursive_layer_args,
        model_name=args.model_load_path,
    )
    if args.dataset == "dyck":
        callback_fn = lambda split: eval_callback_dyck(
            model,
            in_vocab,
            split,
            eval_attachment_decisions=recursive_layer_args.get(
                "attachment_decisions", False
            ),
        )
    elif args.use_stack_tape:
        callback_fn = lambda split: callback_pushdown(
            model,
            in_vocab,
            split,
            data_folder_given=args.dataset,
        )
    else:
        callback_fn = None

    device = torch.device("cuda:{}".format(args.gpu_id))
    model.to(device)
    if args.model_load_path:
        data_collator = collate.VarLengthCollate(None)
        if callback_fn is not None:
            out_val = callback_fn("val")
            out_test = callback_fn("test")
            print("val", out_val)
            print("test", out_test)
        else:
            out = eval_lm(
                interface,
                {key: datasets[key] for key in ["val", "test"]},
                {"val": -1000, "test": -1000},
                device,
                0,
                data_collator,
            )
            print(out)

    else:
        if len(args.save_dir) > 0:
            dir_path = working_dir()
            args.save_dir = os.path.join(dir_path, args.save_dir)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
        eval_keys = ["val", "test"]
        train_loop(
            args,
            interface,
            datasets["train"],
            {key: datasets[key] for key in eval_keys},
            device,
            args.save_dir,
            callback_fn=callback_fn,
            skip_to_step=args.get("skip_to_step", -1),
        )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    main_lm()
