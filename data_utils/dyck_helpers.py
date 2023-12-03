import os
import torch
import string
from tqdm import tqdm
import numpy as np
import argparse
import random
from datasets import Dataset as HFDataset
from vocabulary import WordVocabulary


DATA_DIR = "/u/scr/smurty/unbounded-recursion/data_utils"


def read_dyck_data(splits, vocab_size, max_depth=10):
    def process(sent):
        words = sent.split(" ")[:-1]
        return " ".join(words)

    in_sentences = []
    for split in splits:
        with open(
            "{}/dyck_data/k-{}_d-{}_{}.txt".format(
                DATA_DIR, vocab_size, max_depth, split
            ),
            "r",
        ) as reader:
            sents = [process(line.strip()) for line in reader.readlines()]
            for sent in sents:
                if len(sent.split()) > 500:
                    continue
                in_sentences.append(sent)
    return in_sentences


def get_opening_brackets_prefix(dyck_str):
    """'
    for every closing bracket, return the index of the opening bracket
    """
    dyck_words = dyck_str.split(" ")
    stack = []
    targets = []
    for idx, word in enumerate(dyck_words):
        if "(" in word:
            stack.append(("(", idx))
        else:
            _, oidx = stack.pop()
            targets.append(oidx)
    return targets


def get_all_prefixes(
    dyck_str, target=False, min_dep_length=None, get_opening_idx=False
):
    #### for every prefix that ends in a closing bracket
    dyck_words = dyck_str.split(" ")

    # idxs for all closing brackets
    closing_idxs = [idx for idx, word in enumerate(dyck_words) if ")" in word]
    prefixes = [" ".join(dyck_words[:idx]) for idx in closing_idxs]
    if min_dep_length:
        o_bracket_idxs = get_opening_brackets_prefix(dyck_str)
        chosen_prefixes = [
            idx
            for idx, _ in enumerate(o_bracket_idxs)
            if closing_idxs[idx] - o_bracket_idxs[idx] >= min_dep_length
        ]

        if get_opening_idx:
            o_targets = [o_bracket_idxs[idx] for idx in chosen_prefixes]
            chosen_prefixes = [prefixes[idx] for idx in chosen_prefixes]
            return chosen_prefixes, o_targets
        else:
            return [prefixes[idx] for idx in chosen_prefixes], [
                dyck_words[closing_idxs[idx]] for idx in chosen_prefixes
            ]

    if target:
        # for every closing bracket what was the opening bracket that causes it to close?
        return prefixes, get_opening_brackets_prefix(dyck_str)
    else:
        return prefixes, [dyck_words[idx] for idx in closing_idxs]


def get_identifier_iterator():
    """Returns an iterator to provide unique ids to bracket types."""
    ids = iter(list(string.ascii_lowercase))
    k = 1
    while True:
        try:
            str_id = next(ids)
        except StopIteration:
            ids = iter(list(string.ascii_lowercase))
            k += 1
            str_id = next(ids)
        yield str_id * k


def get_vocab_of_bracket_types(bracket_types):
    """Returns the vocabulary corresponding to the number of brackets.

    There are bracket_types open brackets, bracket_types close brackets,
    START, and END.
    Arguments:
      bracket_types: int (k in Dyck-(k,m))
    Returns:
      Dictionary mapping symbol string  s to int ids.
    """
    id_iterator = get_identifier_iterator()
    ids = [next(id_iterator) for x in range(bracket_types)]
    vocab = {
        x: c
        for c, x in enumerate(
            ["(" + id_str for id_str in ids]
            + [id_str + ")" for id_str in ids]
            + ["START", "END"]
        )
    }
    return vocab, ids


class DyckPDFA:
    """
    Implements a probabilistic finite automata (PFA) that
    generates the dyck language
    """

    def __init__(self, max_stack_depth, bracket_types):
        self.max_stack_depth = max_stack_depth
        self.bracket_types = bracket_types
        self.vocab, self.ids = get_vocab_of_bracket_types(bracket_types)
        self.vocab_list = list(sorted(self.vocab.keys(), key=lambda x: self.vocab[x]))
        self.distributions = {}
        self.list_hash = {}

    def get_token_distribution_for_state(self, state_vec):
        """
        Given a stack state (list of ids, e.g., ['a', 'b']
        produces the probability distribution over next tokens
        """
        if state_vec in self.distributions:
            return self.distributions[state_vec]
        distrib_vec = torch.zeros(len(self.vocab))
        if len(state_vec) == 0:
            for id_str in self.ids:
                distrib_vec[self.vocab["(" + id_str]] = 1 / len(self.ids)
            distrib_vec[self.vocab["END"]] += 1
        elif len(state_vec) == self.max_stack_depth:
            distrib_vec[self.vocab[state_vec[-1] + ")"]] = 1
        else:
            for id_str in self.ids:
                distrib_vec[self.vocab["(" + id_str]] = 1 / len(self.ids)
            distrib_vec[self.vocab[state_vec[-1] + ")"]] = 1
        self.distributions[tuple(state_vec)] = torch.distributions.Categorical(
            distrib_vec / torch.sum(distrib_vec)
        )
        return self.distributions[state_vec]

    def update_state(self, state_vec, new_char_string):
        """
        Updates the DFA state based on the character new_char_string

        For a valid open/close bracket, pushes/pops as necessary.
        For an invalid open/close bracket, leaves state unchanged.
        """
        state_vec = list(state_vec)
        if ")" in new_char_string:
            bracket_type = new_char_string.strip(")")
            if len(state_vec) > 0 and state_vec[-1] == bracket_type:
                state_vec = state_vec[:-1]
        if "(" in new_char_string:
            bracket_type = new_char_string.strip("(")
            if len(state_vec) < self.max_stack_depth:
                state_vec.append(bracket_type)
        return state_vec

    def get_state_hash(self, state_vector):
        hash = []
        for elem in state_vector:
            if "(" in elem:
                hash.append("(")
            else:
                hash.append(")")
        return " ".join(hash)

    def sample(self, length_min, length_max=-1):
        """
        Returns a sample from the Dyck language, as well
        as the maximum number of concurrently-open brackets,
        and the number of times traversed from empty-stack to
        full-stack and back.
        """
        state_vec = []
        string = []
        max_state_len = 0
        empty_full_empty_traversals = 0
        empty_flag = True
        full_flag = False

        stack_path = []
        stack_paths = []
        while True:
            # probs = torch.distributions.Categorical(self.get_token_distribution_for_state(state_vec))
            probs = self.get_token_distribution_for_state(tuple(state_vec))
            new_char = probs.sample()
            new_char_string = self.vocab_list[int(new_char)]
            # Break from generation if END is permitted and sampled
            if new_char_string == "END":
                if len(string) < length_min:
                    continue
                else:
                    string.append(new_char_string)
                    break
            # Otherwise, update the state vector
            string.append(new_char_string)
            state_vec = self.update_state(state_vec, new_char_string)
            if len(state_vec) == 0:
                stack_path_curr = ",".join(stack_path)
                stack_paths.append(stack_path_curr)
                stack_path = []

            else:
                state_hash = self.get_state_hash(state_vec)
                if state_hash not in self.list_hash:
                    self.list_hash[state_hash] = str(len(self.list_hash))
                stack_path.append(self.list_hash[state_hash])

            max_state_len = max(max_state_len, len(state_vec))
            if len(state_vec) == self.max_stack_depth and empty_flag:
                full_flag = True
                empty_flag = False
            if len(state_vec) == 0 and full_flag:
                full_flag = False
                empty_flag = True
                empty_full_empty_traversals += 1
        if len(stack_path) != 0:
            stack_path_curr = ",".join(stack_path)
            stack_paths.append(stack_path_curr)
        return string, max_state_len, empty_full_empty_traversals, stack_paths


def get_training_data(pfsa, min_len, max_len, data_size=200000):
    unique_paths = []
    training_strings = []
    for idx in tqdm(range(data_size)):
        curr_string, _, _, stack_paths = pfsa.sample(min_len, max_len)
        unique_paths += stack_paths
        training_strings.append(curr_string)
    return training_strings, set(unique_paths)


def get_test_data(
    pfsa, min_len, max_len, training_strings, train_paths, data_size=2000
):
    iid_strings = []
    ood_strings = []
    while True:
        if len(ood_strings) == data_size and len(iid_strings) == data_size:
            break
        curr_string, _, _, paths = pfsa.sample(min_len, max_len)
        seen = all([path in train_paths for path in paths])
        if seen and curr_string not in training_strings:
            if len(iid_strings) < data_size:
                iid_strings.append(curr_string)
        elif not seen:
            if len(ood_strings) < data_size:
                ood_strings.append(curr_string)

        if len(ood_strings) % 100 == 0:
            print(len(ood_strings), len(iid_strings))

    return iid_strings, ood_strings


def write_to_file(fname, dataset):
    with open(fname, "w") as writer:
        for dat in dataset:
            curr_string = " ".join(dat)
            writer.write(curr_string)
            writer.write("\n")
    return


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
    return num_merges_before


def add_sos_entries(mat):
    ## mat: N x N
    ### return (N + 1) x (N+1)
    # mat_1 is N x (N+1)
    mat_1 = np.concatenate([np.zeros((len(mat), 1)), mat], axis=1)
    mat_2 = np.concatenate([np.zeros((1, 1 + len(mat))), mat_1], axis=0)
    return mat_2


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


def compute_stack_tape(
    attachment_labels,
):
    num_words = 1 + len(attachment_labels)
    penalty_matrix = np.zeros((num_words, num_words))
    curr_state = np.zeros(num_words)

    ### simulate the shift reduce operations to get depth info
    depth = np.zeros(num_words)
    stack = [0]

    # figure this out...
    for idx in range(1, num_words):
        i = idx - 1
        penalty_matrix[i] = depth
        ### update the stack and depths
        if attachment_labels[i] == idx:
            stack.append([idx])
        else:
            ### this means that the ith token has participated in a reduce operation, so we do reduces.
            curr_constituent = [idx]
            while len(stack) > 1 and attachment_labels[i] not in stack[-1]:
                top = stack.pop()
                curr_constituent = top + curr_constituent
                ### update depth
                for c in curr_constituent:
                    depth[c] += 1

            top = stack.pop()
            curr_constituent = top + curr_constituent
            ### update depth
            for c in curr_constituent:
                depth[c] += 1
            stack.append(curr_constituent)
    penalty_matrix[num_words - 1] = depth
    return (penalty_matrix > 0).astype(np.int32)


def compute_penalty_matrices(dyck_s):
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
    stack_tape = np.stack(
        [
            add_sos_entries(np.cumsum(alpha_matrix, axis=0)),
            add_sos_entries(np.cumsum(beta_matrix, axis=0)),
        ],
        axis=0,
    )
    either = ((stack_tape[0] + stack_tape[1]) > 0).astype("float64")
    either = np.concatenate([np.zeros_like(either[0]).reshape(1, -1), either[:-1]])
    return np.stack([either, either], axis=0)


def build_datasets_dyck(
    vocab=20,
    use_stack_tape=False,
    stack_depth=10,
    data_regime="normal",
    do_compute_stack_labels=True,
):
    def process(sent):
        words = sent.split(" ")[:-1]
        return " ".join(words)

    def read_data(splits):
        in_sentences = []
        index_map = {split: [] for split in splits}
        for split in splits:
            with open(
                "{}/dyck_data/k-{}_d-{}_{}.txt".format(
                    DATA_DIR, vocab, stack_depth, split
                ),
                "r",
            ) as reader:
                sents = [process(line.strip()) for line in reader.readlines()]
                for sent in sents:
                    if len(sent.split()) > 600:
                        continue
                    index_map[split].append(len(in_sentences))
                    in_sentences.append(sent)

        return in_sentences, index_map

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "val", "test"]
    in_sentences, index_map = read_data(splits)
    print("num examples: {}".format(len(in_sentences)))

    in_vocab = WordVocabulary(in_sentences, split_punctuation=False)
    dataset = {}
    for split in splits:
        if data_regime == "small" and split == "train":
            max_sz = 10000
        elif data_regime == "tiny":
            max_sz = 1000
        else:
            max_sz = 100000
        if len(index_map[split]) > max_sz:
            index_map[split] = random.sample(index_map[split], k=max_sz)
        in_subset = get_subset(in_sentences, index_map[split])
        in_subset_tokenized = [in_vocab(s) for s in in_subset]
        in_lens = [len(s) for s in in_subset_tokenized]
        data = {
            "in": in_subset_tokenized,
            "in_len": in_lens,
            "idxs": index_map[split],
        }

        if do_compute_stack_labels:
            # we shift by 1 because the pushdown-LMs makes attachment and next word decisions synchronously!
            data["attachment_labels"] = [
                compute_stack_labels(s)[1][1:] for s in tqdm(in_subset)
            ]

        # only true if we are using pushdown-LMs
        if use_stack_tape:
            data["stack_tape"] = [
                compute_stack_tape(s) for s in tqdm(data["attachment_labels"])
            ]
        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr
    return dataset, in_vocab, in_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=20)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    dyck_pfsa = DyckPDFA(args.max_depth, args.vocab)

    training_strings, paths = get_training_data(dyck_pfsa, 4, 500, data_size=100)
    print(len(training_strings))
