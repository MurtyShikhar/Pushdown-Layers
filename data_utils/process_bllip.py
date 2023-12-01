### tokenize with GPT2 tokenizer
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

import pickle
from tqdm import tqdm
from nltk.tree import Tree


def post_process_op(tree):
    """
    Input: nltk tree where leaf nodes are strings
    Output: nltk tree but use gpt_tokenizer to tokenize leaf nodes and create a subtree corresponding to it.
    """

    def fix(t, is_first, label=None):
        if type(t) == str:
            tokenized = tokenizer.tokenize(t, add_prefix_space=not is_first)
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


def create_bllip(inp_file, out_file):
    with open(inp_file, "r") as f:
        trees = [
            post_process_op(Tree.fromstring(line.strip()))
            for line in tqdm(f.readlines())
        ]

    ### write these nltk trees to file

    with open(out_file, "w") as f:
        for tree in tqdm(trees):
            parse_string = " ".join(str(tree).split())
            f.write(parse_string)
            f.write("\n")


if __name__ == "__main__":
    train = "bllip-lg/prd/train.txt"
    train_out = "bllip-lg-depth/train.txt"
    val = "bllip-lg/prd/val.txt"
    val_out = "bllip-lg-depth/val.txt"
    test = "bllip-lg/prd/test.txt"
    test_out = "bllip-lg-depth/test.txt"

    create_bllip(train, train_out)
    create_bllip(val, val_out)
    create_bllip(test, test_out)
