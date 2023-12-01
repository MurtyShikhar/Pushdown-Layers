'''
    Implements two different beam search variants for pushdown layers:
    1. When pushdown models have a 0/1 stack tape (implicit depth info)
    2. When pushdown models keep depth info in stack tape (explict depth info)

'''
import numpy as np
import torch
import copy

class BeamObj:
    def __init__(self, score, sent_score, attachment_decisions, step):
        self.score = score
        self.sent_score = sent_score
        self.attachment_decisions = attachment_decisions
        self.step = step


def logsumexp(x):
    x = np.array(x)
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))


class BeamSearch:
    def __init__(self, beam_size, input_is_prefix=False):
        self.beam_size = beam_size
        self.input_is_prefix = input_is_prefix

    def update_beam(
        self,
        lm,
        tokenizer,
        beam_curr,
        reduced_set,
        stack_state,
        trafo_state,
        src,
        input_is_prefix = False,
    ):
        """
        Args:
            - lm: the language model
            - tokenizer: the tokenizer
            - beam_curr: list of BeamObj objects
            - reduced_set: list of sets, each set contains the indices that are reduced
            - stack_state: np array of shape BS x step_size
            - trafo_state: the transformer state
            - src: the input sentence
            - input_is_prefix: whether the input is a prefix or not

        Update the beam with the next set of predictions from the LM
        Output:
            - beam_next: list of BeamObj objects
        """
        beam_next = []
        step = beam_curr[0].step
        ### this updates the trafo_state. stack_logprobs is BS x step_size
        with torch.no_grad():
            stack_logprobs, str_logprobs, trafo_state = lm._take_step_with_stack_tape(
                src, trafo_state, stack_state, reduced_set, step
            )
        if input_is_prefix or step not in [len(src[0])-1, len(src[0])-2]:
            beam_next = []
            seen_preds = set()

            for bs, logprob in enumerate(stack_logprobs):
                for curr_idx, curr_attach_logprob in enumerate(logprob):
                    ### if this is not the last step, we can't reduce with the first token
                    if curr_idx == 0:
                        continue
                    if curr_attach_logprob.item() == float("-inf"):
                        continue
                    candidate_score = beam_curr[bs].score + curr_attach_logprob.item()
                    if str_logprobs is not None:
                        candidate_score += str_logprobs[bs].item()

                    candidate_preds = beam_curr[bs].attachment_decisions + [curr_idx]
                    if tuple(candidate_preds) not in seen_preds:
                        seen_preds.add(tuple(candidate_preds))
                        ### keep track of where the candidate originated from
                        beam_next.append(
                            (
                                bs,
                                BeamObj(
                                    candidate_score,
                                    beam_curr[bs].sent_score
                                    + [
                                        (
                                            str_logprobs[bs].item(),
                                            curr_attach_logprob.item(),
                                        )
                                    ],
                                    candidate_preds,
                                    step + 1,
                                ),
                            )
                        )
        elif step == len(src[0]) - 2:
            # this is the second last step i.e. we are predicting stack logits for <eos>. we just hardcode this to be 0!
            beam_next = [
                (
                    origin,
                    BeamObj(
                        beam.score + logprob[0].item() + str_logprobs[origin].item(),
                        beam.sent_score
                        + [(str_logprobs[origin].item(), logprob[0].item())],
                        beam.attachment_decisions + [0],
                        step + 1,
                    ),
                )
                for origin, (beam, logprob) in enumerate(zip(beam_curr, stack_logprobs))
            ]
        elif step == len(src[0]) - 1:
            # for a synchronous model, for the last step i.e. <eos> => </s> mapping, we do not care about the attachment_logits.
            # Thus, no stack_logprobs here, since we don't care about the stack behavior for </s>
            beam_next = [
                (
                    origin,
                    BeamObj(
                        beam.score + str_logprobs[origin].item(),
                        beam.sent_score + [(str_logprobs[origin].item(), 0)],
                        beam.attachment_decisions,
                        step + 1,
                    ),
                )
                for origin, (beam, logprob) in enumerate(
                    zip(beam_curr, stack_logprobs)
                )
            ]
        else:
            raise ValueError("Step is not in the right range!")
            ### otherwise for each beam, we create candidates for every possible reduce operation
        return beam_next, trafo_state

    def init_beam(self):
        return BeamObj(0, [], [], 0)

    def _apply_reduce(self, curr_closed, attachment_decisions, curr_idx, reduced_set):
        for idx, rec_mat in enumerate(curr_closed):
            ### reduce happens if we choose something to reduce that hasnt already been reduced
            if attachment_decisions[idx] != curr_idx:
                red_idx = attachment_decisions[idx]
                rec_mat[red_idx : curr_idx + 1] = 1
                ### everything from red_idx to curr_idx is inaccessible for reduce operations!
                for elem in range(red_idx, curr_idx+1):
                    reduced_set[idx].add(elem)
        return curr_closed, reduced_set


    def update_trafo_state(self, lm, trafo_state, origin_idxs):
        def helper(state_tensor, origin_idxs):
            return torch.stack(
                [copy.deepcopy(state_tensor[idx]) for idx in origin_idxs], dim=0
            )

        ### the +1 is added to store the pre-softmax (final) hidden states!
        new_dict = {
            i: helper(trafo_state.state[i], origin_idxs)
            for i in range(len(lm.trafo.encoder.layers) + 1)
        }
        trafo_state.state = new_dict
        return trafo_state

    def __call__(self, lm, tokenizer, sent, gpu_id):
        device = torch.device("cuda:{}".format(gpu_id))

        ### we start with a beam of size 1
        tokenized_sent = tokenizer(sent)

        beam_curr = [self.init_beam()]
        # nothing is reduced yet
        reduced_state = [set() for b in beam_curr]
        stack_tape = np.zeros((1, len(tokenized_sent)))

        # create a trafo state for each beam entry, so we can advance in parallel
        trafo_state = lm.trafo.encoder.create_state(1, len(tokenized_sent), device)
        for i in range(len(tokenized_sent)):
            if i == len(tokenized_sent) - 1 and self.input_is_prefix:
                break
            src_tensor = torch.tensor([tokenized_sent for _ in beam_curr]).to(device)
            beam_next, trafo_state = self.update_beam(
                lm,
                tokenizer,
                beam_curr,
                reduced_state,
                stack_tape,
                trafo_state,
                src_tensor,
                input_is_prefix = self.input_is_prefix
            )
            # sort the beam entries by score, so the best one is the first one
            beam_next = sorted(beam_next, key=lambda x: x[1].score, reverse=True)[
                : self.beam_size
            ]
            ### first get the stack preds for each beam entry
            stack_pred_set = [b.attachment_decisions[-1] for origin, b in beam_next]
            ### then we need to get the reduced set for the origin of each beam entry
            reduced_state = [
                copy.deepcopy(reduced_state[origin]) for origin, b in beam_next
            ]
            # then we need the stack tape (0/1) for the origin of each beam entry
            stack_tape = np.stack(
                [copy.deepcopy(stack_tape[origin]) for origin, b in beam_next]
            )

            # now we need to update the stack state and reduced state
            stack_tape, reduced_state = self._apply_reduce(
                stack_tape, stack_pred_set, i+1, reduced_state
            )
            ### finally, we need to update trafo state so that trafo state[origin] corresponds to the origin of each beam entry
            trafo_state = self.update_trafo_state(
                lm, trafo_state, [origin for origin, b in beam_next]
            )
            beam_curr = [b for origin, b in beam_next]

        if self.input_is_prefix:
            preds = lm._take_step_with_stack_tape(
                    src_tensor, trafo_state, stack_tape, reduced_state, beam_curr[0].step,
                    return_preds=True
                )
            return [(b.score, b.attachment_decisions, b.sent_score) for b in beam_curr], preds
        else:
            return [(b.score, b.attachment_decisions, b.sent_score) for b in beam_curr]


class BeamSearchDepthBased(BeamSearch):
    def __init__(self, beam_size):
        super().__init__(beam_size)

    def _no_reduce_op(self, stack_pred, step):
        return stack_pred == step + 1

    def _update_reduced_states(self, reduced_state, stack_pred, step):
        ### if stack_pred != step+1, then everything from stack_pred to step is reduced
        for elem in range(stack_pred, step + 1):
            reduced_state.add(elem)

    def _update_stacks(self, reduced_states, stacks, attachment_decisions, step, depths):
        """
        Args:
            - reduced_states: a list of sets, each set contains the indices that are reduced
            - stacks: a list of lists, each list contains the indices that are on the stack
            - attachment_decisions: a list of ints, each int is the index of the token that step wants to reduce with
            - step: the current step
        Returns:
            - updated stacks, reduced_states and depths
        """

        ### this is the last step i.e. we are at <eos> and predict </s>, and we don't care about updating stacks at this point if we are synchronous
        if step == len(depths[0]) - 1:
            return stacks, reduced_states, depths

        for idx, (stack, stack_pred) in enumerate(zip(stacks, attachment_decisions)):
            ### stack is a list of constituents, each constituent is a list of indices
            ### add [step] into stack_state
            if self._no_reduce_op(stack_pred, step):
                stack.append([step + 1])
            else:
                self._update_reduced_states(reduced_states[idx], stack_pred, step)
                curr_constituent = [step + 1]
                while len(stack) > 1 and stack_pred not in stack[-1]:
                    top = stack.pop()
                    curr_constituent = top + curr_constituent
                    for c in curr_constituent:
                        depths[idx][c] += 1
                top = stack.pop()
                curr_constituent = top + curr_constituent
                for c in curr_constituent:
                    depths[idx][c] += 1
                stack.append(curr_constituent)
        return stacks, reduced_states, depths

    def __call__(self, lm, tokenizer, sent, gpu_id, get_surprisal=False):
        device = torch.device("cuda:{}".format(gpu_id))

        ### we start with a beam of size 1, which will grow as we go
        tokenized_sent = tokenizer(sent)

        beam_curr = [self.init_beam()]
        # nothing is reduced yet, and we just a single element on the beam
        reduced_state = [set() for b in beam_curr]
        # maintains the stack depth for each beam entry

        stack_tape = np.zeros((1, len(tokenized_sent)))
        # maintains the actual stack for each beam entry - needed for updating the stack depth

        # if we are synchronous, we start by pushing in <s> into the stack
        stacks = [[[0]] for _ in beam_curr]

        # create a trafo state for each beam entry, so we can advance in parallel
        trafo_state = lm.trafo.encoder.create_state(1, len(tokenized_sent), device)

        if get_surprisal:
            word_logprobs = []
            best_incremental_parse = []
            total_logprob_so_far = 0.0

        for i in range(len(tokenized_sent)):
            src_tensor = torch.tensor([tokenized_sent for _ in beam_curr]).to(device)
            beam_next, trafo_state = self.update_beam(
                lm,
                tokenizer,
                beam_curr,
                reduced_state,
                stack_tape,
                trafo_state,
                src_tensor,
            )
            # sort the beam entries by score, so the best one is the first one
            beam_next = sorted(beam_next, key=lambda x: x[1].score, reverse=True)
            if get_surprisal:
                # marginalize over all beam scores, noting that beam_next considers all possible extensions of beam_curr
                # i.e. all attachment decision for the word that was just predicted! The stack probs for these aren't part of surprisal
                # at this token, but will be part of surprisal at the next token
                curr_origins = list(set([origin for origin, beam in beam_next]))
                curr_logprobs = [
                    np.sum(beam_next[origin][1].sent_score[:i])
                    + beam_next[origin][1].sent_score[i][0]
                    for origin in curr_origins
                ]
                logprobs_new = logsumexp(curr_logprobs)
                best_incremental_parse.append(
                    (beam_curr[0].score, beam_curr[0].attachment_decisions)
                )
                word_logprobs.append(logprobs_new - total_logprob_so_far)
                total_logprob_so_far = logprobs_new

            beam_next = beam_next[: self.beam_size]

            # first get the stack preds for each beam entry
            stack_pred_set = [b.attachment_decisions[-1] for origin, b in beam_next]
            # then we need to get the reduced set for the origin of each beam entry
            reduced_state = [
                copy.deepcopy(reduced_state[origin]) for origin, b in beam_next
            ]

            # then we need the stack *depth* state for the origin of each beam entry
            stack_tape = np.stack(
                [copy.deepcopy(stack_tape[origin]) for origin, b in beam_next]
            )

            # then we need the actual stacks for the origin of each beam entry
            stacks = [copy.deepcopy(stacks[origin]) for origin, b in beam_next]

            # now we need to update the stacks, reduced state, and stack depth state
            stacks, reduced_state, stack_tape = self._update_stacks(
                reduced_state, stacks, stack_pred_set, i, stack_tape
            )
            # finally, we need to update trafo state so that trafo state[origin] corresponds to the origin of each beam entry
            trafo_state = self.update_trafo_state(
                lm, trafo_state, [origin for origin, b in beam_next]
            )
            beam_curr = [b for origin, b in beam_next]

        if get_surprisal:
            return (
                word_logprobs,
                best_incremental_parse,
                [(b.score, b.attachment_decisions, b.sent_score) for b in beam_curr],
            )
        return [(b.score, b.attachment_decisions, b.sent_score) for b in beam_curr]
