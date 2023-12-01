from vocabulary import WordVocabulary
from datasets import Dataset as HFDataset
import sequence
import pickle
import random
from collections import Counter
import numpy as np
from tqdm import tqdm
from nltk import Tree


def flatten(parse, add_eos):
    def helper(p):
        if type(p) == str:
            return p
        else:
            return " ".join(helper(x) for x in p)

    if type(parse) == Tree:
        words = " ".join(parse.leaves())
    else:
        words = helper(parse)
    if add_eos:
        return "{} <eos>".format(words)
    else:
        return words


def attachment_decisions_to_tree(stack_pred, sent):
    ### stack_pred is a list of stack_predictions. convert this into a tree

    stack = []
    stack_idxs = []

    ### add <s> manually!
    sent_words = ["<s>"] + sent.split(" ")

    for idx, pred in enumerate(stack_pred):
        if pred == idx:
            stack.append(sent_words[idx])
            stack_idxs.append(idx)
        else:
            ### perform a bunch of reduce operations till the top of the stack is pred
            if sent_words[idx] == "<eos>":
                constituent = None
            else:
                constituent = sent_words[idx]
            while len(stack_idxs) > 1 and stack_idxs[-1] != pred:
                top = stack.pop()
                stack_idxs.pop()
                if constituent is None:
                    constituent = top
                else:
                    constituent = (top, constituent)
            ### at this point stack_idxs[-1] == pred and stack[-1] == sent_words[pred]
            stack_idxs.pop()
            constituent = (stack.pop(), constituent)
            stack.append(constituent)
            stack_idxs.append(idx)
    assert len(stack) == 1
    return stack[0][-1]


def get_elems(list_of_list):
    ### flatten out a list of lists
    ret = []
    for l in list_of_list:
        if type(l) == list:
            ret.extend(get_elems(l))
        else:
            ret.append(l)
    return ret




def compute_stack_tape(
    attachment_labels,
    head_info,
    type_labels=None,
    with_depth_info=False,
):
    ### the stack_tape is a matrix of size len(attachment_labels) x len(attachment_labels) where the (i, j) entry is 1 if
    ### after observing the input string till position i-1, the jth token has either participated in a reduce operation.
    ### given the stack labels for an input string, compute the penalty matrices for the input string
    ### for example, if the input_str is "a b c d" and the parse is "((a b) (c d))", then the stack labels are
    ### [0, 0, 2, 1] and the penalty matrix is [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]

    num_words = len(attachment_labels)
    penalty_matrix = np.zeros((num_words, num_words))
    curr_state = np.zeros(num_words)

    ### simulate the shift reduce operations to get depth info
    if with_depth_info:
        depth = np.zeros(num_words)
        stack = []

    for i in range(num_words):
        if with_depth_info:
            penalty_matrix[i] = depth
        else:
            penalty_matrix[i] = curr_state
        ### update the stack and depths
        if with_depth_info:
            if attachment_labels[i] == i:
                stack.append([i])
            else:
                ### this means that the ith token has participated in a reduce operation, so we do reduces.
                curr_constituent = [i]
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
        ### update curr_state info
        ### if attachment_labels[i] != i, then the ith token has participated in a reduce operation
        if attachment_labels[i] != i:
            if i == num_words - 1:
                start = attachment_labels[i]
            else:
                start = head_info[i - 1] + 1
            curr_state[start : i + 1] = 1 if type_labels is None else type_labels[i]
        elif type_labels is not None:
            curr_state[i : i + 1] = type_labels[i]
    return np.concatenate([penalty_matrix[1:], penalty_matrix[-1][None, :]], axis=0)


def get_shift_reduce_actions(parse):
    """
    given an input_str and its corresponding parse as a tuple of tuples, return the series of shift reduce operations that would produce the parse
    Examples:
    if the input_str is "a b c d" and the parse is "((a, b), (c, d))", then the shift reduce operations are
    shift a, shift b, reduce, shift c, shift d, reduce, reduce
    if the input_str is "a b c d" and the parse is "(a, (b, (c, d)))", then the shift reduce operations are
    shift a, shift b, shift c, shift d, reduce reduce reduce
    if the input_str is "a b c d" and the parse is "(((a, b), c), d)", then the shift reduce operations are
    shift a, shift b, reduce, shift c, reduce, shift d, reduce
    """

    actions = []

    def shift_reduce_recursive(parse):
        if type(parse) == str or len(parse) == 1:
            actions.append("shift")
        else:
            for p in parse:
                shift_reduce_recursive(p)
            actions.append("reduce")

    shift_reduce_recursive(parse)
    return actions


def accomodate_sos_and_eos(attachment_labels, constituent_labels=None):
    ### the first 0 is for the SOS token and the last 0 is for the EOS token
    if not constituent_labels:
        return [0] + [1 + label for label in attachment_labels] + [0], None
    else:
        return [0] + [1 + label for label in attachment_labels] + [0], ["SOS"] + [
            label for label in constituent_labels
        ] + ["EOS"]




def get_constituents(penalty_matrix, attachment_labels):
    """
    At each point what are the current constituents in the stack?
    Represent the constituent as a list of indices that are part of the constituent.

    Example:
    if the input_str is "a b c d" and the parse is "((a b) (c d))", then the stack labels are
    [0, 0, 2, 1] and the penalty matrix is [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]

    At the first step, the constituents are [0]
    At the second step, the constituents are [[0, 1]]
    At the third step, the constituents are [[0, 1], [2]]
    Finally, the constituents are [[0, 1], [2, 3]]

    Once we have these constituents we can replace stack labels with a one hot instead, so that the model may
    choose its own headwords for a given constituent.
    """

    num_words = len(attachment_labels)
    constituents = []
    curr_constituent = []
    for i in range(num_words):
        if penalty_matrix[i][i] == 1:
            curr_constituent.append(i)
        else:
            curr_constituent.append(i)
            constituents.append(curr_constituent)
            curr_constituent = []

    return constituents


def get_constituent_labels(tree_tuple):
    split_vals = {}

    def get_len(t, st, label=None):
        if type(t) == str:
            split_vals[(st, st)] = label
            return 1
        elif len(t) == 1:
            return get_len(t[0], st, label=t.label())
        else:
            curr_len = 0
            for c in t:
                l1_len = get_len(c, st + curr_len)
                curr_len += l1_len
            split_vals[(st, st + curr_len - 1)] = t.label()
            return curr_len

    get_len(tree_tuple, 0)
    return split_vals


def compute_attachment_labels_text(parse, input_str, with_depth_info=False):
    """
    given an input_str and its corresponding parse, return for each token in the input_str the earliest token
    in the input_str that the token wants to reduce with.
    Examples:
    if the input_str is "a b c d" and the parse is "((a b) (c d))", then the stack labels are
    [0, 0, 2, 1] because the series of shift reduce operations that would produce the parse is
    shift a (0), shift b (1), reduce (2), shift c (3), shift d (4), reduce (5), reduce (6)
    if the input_str is "a b c d" and the parse is "(a, (b, (c, d)))", then the stack labels are
    [0, 1, 2, 0] because the series of shift reduce operations that would produce the parse is
    shift a (0), shift b (1), shift c (2), shift d (3), reduce (4),  reduce (5), reduce (6)
    if the input_str is "a b c d" and the parse is "(((a, b), c), d)", then the stack labels are
    [0, 0, 1, 2] because the series of shift reduce operations that would produce the parse is
    shift a (0), shift b (1), reduce (2), shift c (3), reduce (4), shift d (5), reduce (6)

    if with_depth_info, then along with type labels, return the number of reduce operations that have been performed.
    """

    stack_actions = get_shift_reduce_actions(parse)

    if type(parse) == Tree:
        constituent_labels_given = get_constituent_labels(parse)
        constituent_labels = []
    attachment_labels = []
    stack = []

    curr_word_idx = 0
    stack_idx = 0
    cstart = {}

    if "<eos>" in input_str:
        ### for reasons, we added a eos token to the input string
        num_words = len(input_str.split(" ")) - 1  #### remove one for the EOS token!!!!
    else:
        num_words = len(input_str.split(" "))
    while curr_word_idx < num_words:
        next_shift = stack_idx + 1

        stack_top = None
        while next_shift < len(stack_actions) and stack_actions[next_shift] != "shift":
            stack_top = stack.pop()
            next_shift += 1

        if stack_top is None:
            attachment_labels.append(curr_word_idx)
            constituent_labels.append(
                constituent_labels_given[(curr_word_idx, curr_word_idx)]
            )
        else:
            ### everything from head[stack_top] to curr_word_idx is a constituent.
            ### also get the constituent label for the constituent
            attachment_labels.append(stack_top)
            if stack_top not in cstart:
                constituent_start = stack_top
            else:
                constituent_start = cstart[stack_top]
            constituent_labels.append(
                constituent_labels_given[(constituent_start, curr_word_idx)]
            )

            cstart[curr_word_idx] = constituent_start

        stack.append(curr_word_idx)
        stack_idx = next_shift
        curr_word_idx += 1

    attachment_labels, type_labels = accomodate_sos_and_eos(
        attachment_labels, constituent_labels
    )

    if type_labels is not None:
        return {
            "type_labels": type_labels,
            "attachment_labels": attachment_labels,
            "cstart_info": cstart,
        }
    else:
        return {
            "attachment_labels": attachment_labels,
            "cstart_info": cstart,
        }


def binarize_tree(parse):
    if type(parse) == str:
        return parse
    else:
        if len(parse) == 1:
            return binarize_tree(parse[0])
        else:
            return (binarize_tree(parse[0]), binarize_tree(parse[1:]))


def build_datasets_pushdown(
    use_stack_tape=False,
    data_regime="normal",
    do_compute_attachment_labels=True,
    only_vocab=False,
    data_file_given=None,
    with_depth_info=False,
):
    def read_data(splits):
        in_sentences = []
        parses = []
        index_map = {split: [] for split in splits}
        for split in splits:
            split_file = split

            if not data_file_given:
                data_file = "bllip-lg-depth"
            else:
                data_file = data_file_given

            with open(
                "{}/{}/{}.txt".format(
                    "/u/scr/smurty/unbounded-recursion/data_utils",
                    data_file,
                    split_file,
                ),
                "r",
            ) as reader:
                print("Reading trees for {}".format(split_file))
                data = [
                    Tree.fromstring(l.strip()) for l in tqdm(reader.readlines())
                ]
            for sent in tqdm(data):
                index_map[split].append(len(in_sentences))
                in_sentences.append(flatten(sent, add_eos=True))
                if not isinstance(sent, Tree):
                    parses.append(binarize_tree(sent))
                else:
                    parses.append(sent)
        return in_sentences, parses, index_map

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    def process_with_stack_type_vocab(type_labels, vocab):
        return [vocab[t] for t in type_labels]

    splits = ["train", "val", "test"]
    ### NOTE: if math, sent and parses are the same.
    in_sentences, parses, index_map = read_data(splits)
    print("num examples: {}".format(len(in_sentences)))

    in_vocab = WordVocabulary(in_sentences, split_punctuation=False)
    if only_vocab:
        return in_vocab


    dataset = {}
    for split in splits:
        print("Processing {} data".format(split))
        if data_regime == "small" and split == "train":
            max_sz = 10000
        elif data_regime == "tiny":
            max_sz = 1000
        else:
            max_sz = len(index_map[split])
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

        stack_info_dict_list = [
            compute_attachment_labels_text(parses[idx], in_sentences[idx])
            for idx in tqdm(index_map[split])
        ]
        if do_compute_attachment_labels:
            # we synchronously predict both the next word and the stack label corresponding to the next word!
            ### <s> => model predicts "The" and attachment decision for "The"
            ### <s> The => model predicts "man" and attachment decision for "man"
            ### <s> The man => model predicts "likes" and attachment decision for "likes"
            ### <s> The man likes => model predicts "apples" and attachment decision for "apples"
            ### <s> The man likes apples => model predicts <eos> and attachment decision for <eos>
            ### <s> The man likes apples <eos> => model predicts </s> and attachment decision for </s>

            # i.e. for <s> The man likes apples <eos>, we compute 1 (for the),  1 (for man), 3 (for likes), 2 (for apples), 0 (for <eos>)
            data["attachment_labels"] = [
                l["attachment_labels"][1:] for l in stack_info_dict_list
            ]

        if use_stack_tape:
            data["stack_tape"] = [
                compute_stack_tape(
                    stack_info_dict_list[idx]["attachment_labels"],
                    stack_info_dict_list[idx]["cstart_info"],
                    type_labels=None,
                    with_depth_info=with_depth_info,
                )
                for idx, stack_info_dict in tqdm(enumerate(stack_info_dict_list))
            ]

        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr

    return dataset, in_vocab, in_sentences
