from transformer_helpers import create_model_interface, create_lm
import torch
from transformers import AutoTokenizer, RobertaForMaskedLM
from scipy.spatial import distance
import random
import numpy as np
import torch
import random
import collate

import layers

from tqdm import tqdm

import torch.nn.functional as F

from omegaconf import OmegaConf, open_dict

def convert_tree_to_tuple(tree):
    """Convert NLTK tree to a tuple representation. """
    def fix(t):
        if type(t) == str:
            return t
        elif len(t) == 1:
            return fix(t[0])
        else:
            all_children = [c for c in t]
            all_children = [c for c in t]
            return (fix(all_children[0]), fix(tuple(all_children[1:])))

    return fix(tree)

def get_base_transformer_lm(
    args, in_vocab, recursive_layer_args, model_name=None
):
    if args.get("with_depth_info", False):
        try:
            with open_dict(recursive_layer_args):
                recursive_layer_args["stack_type_vocab"] = args.get("max_depth", 50)
        except:
            recursive_layer_args["stack_type_vocab"] = args.get("max_depth", 50)

    model = create_lm(
        args,
        len(in_vocab),
        args.vec_dim,
        args.n_heads,
        args.encoder_n_layers,
        ff_multiplier=args.get("ff_multiplier", 1),
        use_stack_tape=args.use_stack_tape,
        recursive_layer_args=recursive_layer_args,
    )
    if model_name:
        print("loading pretrained model from {}".format(model_name))
        checkpoint = torch.load(model_name, map_location=torch.device("cpu"))
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)


    interface = create_model_interface(
        model,
        in_vocab=in_vocab,
        is_lm=True,
    )
    return model, interface


def is_same_type(w_open, w_close):
    if type(w_open) != str:
        return False
    if "(" not in w_open or ")" not in w_close:
        return False
    else:
        return w_open[-1] == w_close[0]


def get_stack_info(dyck_s):
    words = dyck_s.split(" ")
    stack = []
    num_merges_before = []
    curr_merges = 0
    for word in words:
        if "(" in word:
            num_merges_before.append(curr_merges)
            stack.append(word)
        elif ")" in word:
            ### reduce everything at the top of the stack?
            merged = None
            while not is_same_type(stack[-1], word):
                top = stack.pop()
                if merged is None:
                    merged = top
                else:
                    merged = (top, merged)
                    curr_merges += 1

            assert is_same_type(stack[-1], word)
            top = stack.pop()
            if merged is None:
                merged = (top, word)
                curr_merges += 1

            else:
                merged = ((top, merged), word)
                curr_merges += 2
            num_merges_before.append(curr_merges)
            stack.append(merged)
    # The [0] is added for the BOS token
    return num_merges_before


def add_sos_entries(mat):
    ## mat: N x N
    ### return (N + 1) x (N+1)
    # mat_1 is N x (N+1)
    mat_1 = np.concatenate([np.zeros((len(mat), 1)), mat], axis=1)
    mat_2 = np.concatenate([np.zeros((1, 1 + len(mat))), mat_1], axis=0)
    return mat_2


def compute_penalty_matrices(dyck_s, as_bias=False):
    stack_info = get_stack_info(dyck_s)
    str_len = len(stack_info)
    alpha_matrix = np.zeros((str_len, str_len))
    beta_matrix = np.zeros((str_len, str_len))
    num_reduces_curr = 0
    curr_stack = []
    for idx, total_reduces in enumerate(stack_info):
        curr_stack.append([idx])
        while num_reduces_curr < total_reduces:
            for jdx in curr_stack[-2]:
                alpha_matrix[idx][jdx] += 1
            for jdx in curr_stack[-1]:
                beta_matrix[idx][jdx] += 1
            x1 = curr_stack.pop()
            x2 = curr_stack.pop()
            curr_stack.append(x1 + x2)
            num_reduces_curr += 1
    stack_repr = np.stack(
        [
            add_sos_entries(np.cumsum(alpha_matrix, axis=0)),
            add_sos_entries(np.cumsum(beta_matrix, axis=0)),
        ],
        axis=0,
    )
    if as_bias:
        either = ((stack_repr[0] + stack_repr[1]) > 0).astype("float64")
        either = np.concatenate([np.zeros_like(either[0]).reshape(1, -1), either[:-1]])
        return np.stack([either, either], axis=0)
    else:
        return stack_repr


def run_lm_decoding(tokenizer, lm, prefixes, gpu_id):
    data_collator = collate.VarLengthCollate(None)
    max_decoding_steps = 50

    # the tokenizer just adds an <SOS> token to the front of the string
    def tokenizer_helper(inp_slice):
        inp_list = [tokenizer(s) for s in inp_slice]
        in_lens = [len(s) for s in inp_list]

        inp_to_collate = [{"in": x, "in_len": y} for x, y in zip(inp_list, in_lens)]
        inp = data_collator(inp_to_collate)
        in_len = inp["in_len"].long()
        return inp["in"].transpose(0, 1), in_len

    batch_size = 128
    st = 0
    device = torch.device("cuda:{}".format(gpu_id))
    decoded_sents = []
    while st < len(prefixes):
        en = min(len(prefixes), st + batch_size)
        cslice = prefixes[st:en]
        inputs, input_lens = tokenizer_helper(cslice)
        inputs = inputs.to(device)
        input_lens = input_lens.to(device)
        with torch.no_grad():
            outputs, stack_out = lm.run_greedy(inputs, input_lens, max_decoding_steps)
            preds = outputs["data"].argmax(axis=-1)
            out_lens = outputs["length"]
            for pred, out_len in zip(preds, out_lens):
                decoded_sents.append(pred[:out_len].tolist())
        st = en
    return decoded_sents, stack_out






def compute_perplexity(lm, str_logits, inputs, input_lens):
    # (len x bs)
    targets = inputs[:, 1:].transpose(0, 1)
    target_lens = input_lens - 1

    # (len x bs x vocab)
    str_logits = str_logits[:-1, :]  # remove the last token

    # compute length mask (len x bs)
    len_mask = ~lm.generate_len_mask(targets.shape[0], target_lens).transpose(0, 1)

    # compute the loss
    loss_curr = layers.cross_entropy(str_logits, targets, reduction="none")
    loss_curr = loss_curr.reshape_as(targets) * len_mask
    return loss_curr.sum().cpu().numpy(), len_mask.sum().item()


def make_preds_stack_teacher_force_x(tokenizer, lm, prefixes, gpu_id, is_dyck=True):
    data_collator = collate.VarLengthCollate(None)

    batch_size = 64
    st = 0
    device = torch.device("cuda:{}".format(gpu_id))
    final_logprobs = []
    all_str_logits = []

    loss = 0.0
    total = 0.0
    # the tokenizer just adds an <SOS> token to the front of the string
    def tokenizer_helper(inp_slice):
        inp_list = [tokenizer(s) for s in inp_slice]
        in_lens = [len(s) for s in inp_list]

        inp_to_collate = [{"in": x, "in_len": y} for x, y in zip(inp_list, in_lens)]
        inp = data_collator(inp_to_collate)
        in_len = inp["in_len"].long()
        return inp["in"].transpose(0, 1), in_len

    with tqdm(total=len(prefixes)) as progress_bar:
        while st < len(prefixes):
            en = min(len(prefixes), st + batch_size)
            cslice = prefixes[st:en]
            inputs, input_lens = tokenizer_helper(cslice)
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            with torch.no_grad():
                logprobs_curr, all_str_logits_curr = lm.run_greedy_with_stack(
                    inputs, input_lens, get_str_logits=True
                )
                final_logprobs += logprobs_curr
                loss_curr, total_curr = compute_perplexity(
                    lm, all_str_logits_curr, inputs, input_lens
                )
                loss += loss_curr
                total += total_curr

            progress_bar.update(en - st)
            st = en
    final_prob = torch.stack(final_logprobs, dim=0)
    return F.softmax(final_prob, dim=1), np.exp(loss / total)


def get_attn_flows(attn_list, bs):
    attn_flow = [attn_list[0][idx] for idx in range(bs)]
    for attn_mat in attn_list[1:]:
        attn_flow = [torch.matmul(attn_mat[idx], attn_flow[idx]) for idx in range(bs)]
    return attn_flow


def get_average_attn(attn_list, bs, layer):
    if layer != -1:
        return [attn_list[layer][idx] for idx in range(bs)]
    else:
        attn_avg = [attn_list[0][idx] for idx in range(bs)]
        for attn_mat in attn_list[1:]:
            attn_avg = [attn_avg[idx] + attn_mat[idx] for idx in range(bs)]
        return [x / len(attn_list) for x in attn_avg]



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accomodate_sos(label_array):
    st_len = len(label_array)
    return [0] + [p + 1 for p in label_array[:-1]] + [0]


def convert_to_softmax_labels(label_array):
    sz = len(label_array)
    out = np.zeros((sz, sz))
    for idx, elem in enumerate(label_array):
        elem = int(elem)
        if elem != 0:
            out[elem][idx] = 1.0
    for idx, elem in enumerate(out):
        if elem.sum() == 0:
            out[idx][idx] = 1.0
    return out.argmax(axis=1)


def compute_stack_labels(dyck_s):
    stack_info = get_stack_info(dyck_s)
    str_len = len(stack_info)
    label_matrix = np.zeros((str_len, str_len))
    num_reduces_curr = 0
    curr_stack = []
    available = set([idx for idx in range(str_len)])
    label_array = np.zeros(str_len)
    for idx, total_reduces in enumerate(stack_info):
        curr_stack.append([idx])
        while num_reduces_curr < total_reduces:
            second_stack = curr_stack[-2]
            ### set some entries in label_matrix[idx] to 1
            for elem in second_stack:
                if elem in available:
                    label_matrix[idx][elem] = 1
                    if len(second_stack) == 1:
                        label_array[elem] = idx
                        available.remove(elem)
            top = curr_stack.pop()
            second = curr_stack.pop()
            curr_stack.append(second + top)
            num_reduces_curr += 1

    ### everything still unreduced gets reduced at the final stage.
    for elem in range(str_len - 1):
        if elem in available:
            label_array[elem] = -1  # str_len - 1
    ### TO accomodate SOS token, add 1 to everything
    return label_matrix, convert_to_softmax_labels(accomodate_sos(label_array))


if __name__ == "__main__":
    s = "(a (b (c (d (e e) d) (g g) c) b) a)"
    out = compute_stack_labels(s)
    print(out[1])
    print(s.split(" "), len(s.split(" ")))
    sz = len(out[1])
    o2 = np.zeros((sz, sz))
    for idx, elem in enumerate(out[1]):
        elem = int(elem)
        if elem != 0:
            o2[elem][idx] = 1.0
    for idx, elem in enumerate(o2):
        if elem.sum() == 0:
            o2[idx][idx] = 1.0
    print(np.argmax(o2, axis=1))
    # print(o2)
