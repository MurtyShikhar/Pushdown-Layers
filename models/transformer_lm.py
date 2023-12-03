import torch
import random
import math
from layers import Transformer, TiedEmbedding, PositionalEncoding
from typing import Callable, Optional
import numpy as np
import torch.nn.functional as F


class DotDict(dict):
    def __getattr__(self, item):
        if item not in self:
            raise AttributeError
        return self.get(item)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TransformerResult(DotDict):
    data: torch.Tensor
    length: torch.Tensor

    @staticmethod
    def create(data: torch.Tensor, length: torch.Tensor):
        return TransformerResult({"data": data, "length": length})


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        n_input_tokens: int,
        state_size: int = 512,
        ff_multiplier: float = 1,
        max_len: int = 5000,
        embedding_dropout: float = -1.0,
        output_dropout: float = -1.0,
        transformer=Transformer,
        tied_embedding: bool = False,
        pos_embedding: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        encoder_sos: bool = True,
        embedding_init: str = "pytorch",
        scale_mode: str = "none",
        **kwargs
    ):
        """
        :param n_input_tokens: Number of channels for the input vectors
        :param n_out_tokens: Number of channels for the output vectors
        :param state_size: The size of the internal state of the transformer
        """
        super().__init__()

        assert scale_mode in ["none", "opennmt", "down"]
        assert embedding_init in ["pytorch", "xavier", "kaiming"]

        self.tied_embedding = tied_embedding

        self.encoder_eos = n_input_tokens
        self.encoder_sos = n_input_tokens + 1 if encoder_sos else None
        self.state_size = state_size
        self.embedding_init = embedding_init
        self.ff_multiplier = ff_multiplier
        self.n_input_tokens = n_input_tokens
        self.scale_mode = scale_mode
        self.pos = pos_embedding or PositionalEncoding(
            state_size,
            max_len=max_len,
            batch_first=True,
            scale=(1.0 / math.sqrt(state_size)) if scale_mode == "down" else 1.0,
        )

        self.register_buffer("int_seq", torch.arange(max_len, dtype=torch.long))
        self.construct(transformer, **kwargs)
        self.reset_parameters()
        # need this flag for the training loop helpers
        if embedding_dropout != -1.0:
            self.embedding_dropout = torch.nn.Dropout(embedding_dropout)
        else:
            self.embedding_dropout = None
        if output_dropout != -1.0:
            self.output_dropout = torch.nn.Dropout(output_dropout)
        else:
            self.output_dropout = None
        self.mode = "lm"
        self.synchronous = False

    def construct(self, transformer, **kwargs):
        self.input_embedding = torch.nn.Embedding(
            self.n_input_tokens + 1 + int(self.encoder_sos is not None),
            self.state_size,
        )

        if self.tied_embedding:
            self.output_map = TiedEmbedding(self.input_embedding.weight)
        else:
            self.output_map = torch.nn.Linear(
                self.state_size,
                self.n_input_tokens + 1 + int(self.encoder_sos is not None),
            )

        self.trafo = transformer(
            d_model=self.state_size,
            dim_feedforward=int(self.ff_multiplier * self.state_size),
            **kwargs
        )

    def get_offsets(self):
        def get_val(obj):
            if type(obj) == float:
                return obj
            elif type(obj) == torch.nn.modules.sparse.Embedding:
                return (obj.weight[0].tolist(), obj.weight[1].tolist())
            elif obj.ndim == 0:
                return obj.item()
            else:
                return obj.tolist()

        if self.trafo.use_stack_tape:
            if self.trafo.per_layer_offsets:
                alpha_ret = [
                    get_val(layer.self_attn.alpha)
                    for layer in self.trafo.encoder.layers
                ]
                beta_ret = [
                    get_val(layer.self_attn.beta) for layer in self.trafo.encoder.layers
                ]
            else:
                alpha_ret = get_val(self.trafo.alpha)
                beta_ret = get_val(self.trafo.beta)
            return alpha_ret, beta_ret
        else:
            return 0.0, 0.0

    def input_embed(self, x: torch.Tensor) -> torch.Tensor:
        src = self.input_embedding(x.long())
        return src

    def reset_parameters(self):
        if self.embedding_init == "xavier":
            torch.nn.init.xavier_uniform_(self.input_embedding.weight)
        elif self.embedding_init == "kaiming":
            torch.nn.init.kaiming_normal_(self.input_embedding.weight)
        if not self.tied_embedding:
            torch.nn.init.xavier_uniform_(self.output_map.weight)

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[:max_len] >= len.unsqueeze(-1)

    def run_teacher_forcing(
        self, src: torch.Tensor, src_len: torch.Tensor, stack_tape
    ) -> TransformerResult:
        in_len_mask = self.generate_len_mask(src.shape[1], src_len)
        res, attachment_logits = self.trafo(
            src, tgt=None, src_length_mask=in_len_mask, stack_tape=stack_tape
        )

        if self.output_dropout is not None:
            res = self.output_dropout(res)

        ret_dict = {"output": TransformerResult.create(self.output_map(res), src_len)}
        if attachment_logits is not None:
            ret_dict["attachment_logits"] = attachment_logits
        return ret_dict

    def pos_embed(self, t: torch.Tensor, offset: int) -> torch.Tensor:
        if self.scale_mode == "opennmt":
            t = t * math.sqrt(t.shape[-1])

        return self.pos(t, offset)

    def get_encoder_layers(self):
        return self.trafo.num_encoder_layers

    def get_attention_matrices(self, src, src_len, stack_tape=None):
        mask = self.generate_len_mask(src.shape[1], src_len)
        src = self.pos_embed(self.input_embed(src), 0)
        attn_matrices = self.trafo.get_attn_matrices(
            src, tgt=None, stack_tape=stack_tape, src_length_mask=mask
        )
        mats = []
        for mat in attn_matrices:
            curr_mats = []
            for clen, batch_obj in zip(src_len, mat):
                curr_att_mat = batch_obj[:clen, :clen].cpu().detach().numpy()
                curr_mats.append(curr_att_mat)
            mats.append(curr_mats)
        return mats

    def get_attention_sparsity(self, src, src_len):
        mask = self.generate_len_mask(src.shape[1], src_len)
        src = self.pos_embed(self.input_embed(src), 0)
        attn_matrices = self.trafo.get_attn_matrices(
            src, tgt=None, src_length_mask=mask
        )

        total_entropy = 0.0
        for mat in attn_matrices:
            for clen, batch_obj in zip(src_len, mat):
                curr_att_mat = batch_obj[:clen, :clen]
                for idx, attns in enumerate(curr_att_mat):
                    total_entropy += torch.distributions.Categorical(
                        attns[: idx + 1]
                    ).entropy()
        return total_entropy / len(attn_matrices)

    def encoder_only(self, src, mask, layer_id=-1, gaussian_noise=None):
        src = self.pos_embed(self.input_embed(src), 0)
        if gaussian_noise is not None:
            src += gaussian_noise

        return self.trafo.get_hidden_states(src, mask, layer_id=layer_id, is_lm=True)

    def _apply_reduce(self, curr_closed, attachment_decisions, curr_idx, reduced_set):
        for idx, rec_mat in enumerate(curr_closed):
            ### reduce happens if we choose something to reduce that hasnt already been reduced
            if attachment_decisions[idx] != curr_idx:
                red_idx = attachment_decisions[idx]
                rec_mat[red_idx : curr_idx + 1] = 1
                ### everything from red_idx to curr_idx-1 is inaccessible for reduce operations!
                for elem in range(red_idx, curr_idx):
                    reduced_set[idx].add(elem)
        return curr_closed, reduced_set

    def _take_step_with_stack_tape(
        self, src, trafo_state, stack_state, list_of_reduced, step
    ):
        ### src: bs x seq_len

        next_tgt = torch.cat(
            [
                self.pos_embed(
                    self.input_embed(inp[step : step + 1]).unsqueeze(1), step
                )
                for inp in src
            ]
        )

        ## src_length_mask: bs x (step+1)
        src_length_mask = (
            torch.tensor([False] * step + [False])
            .unsqueeze(1)
            .transpose(0, 1)
            .to(src.device)
        )
        output, attachment_logits = self.trafo.encoder.one_step_forward(
            trafo_state,
            next_tgt,
            src_length_mask=src_length_mask,
            stack_tape=torch.tensor(stack_state).unsqueeze(1).to(src.device),
        )

        preds = self.output_map(output[:, 0])
        if step != src.shape[1] - 1:
            targets = src[:, step + 1]  # bs
        else:
            ### this is the last step, so the target is to generate the end token
            targets = torch.tensor([self.encoder_eos] * src.shape[0]).to(src.device)

        pred_logprob = F.log_softmax(preds, dim=-1)  # bs x vocab_size
        ## get log prob of targets
        str_logprobs = torch.gather(pred_logprob, 1, targets.unsqueeze(1)).squeeze(1)
        ### logits_curr_t: bs x (step + 1)
        logits_curr_t = attachment_logits[0][:, 0, :]
        logprobs_curr = F.log_softmax(logits_curr_t, dim=-1)

        ### logprobs corresponding to things that have been reduced should be -inf
        for bs, reduced_set in enumerate(list_of_reduced):
            for elem in reduced_set:
                logprobs_curr[bs][elem] = float("-inf")

        return logprobs_curr, str_logprobs, trafo_state

    def run_greedy_with_stack(
        self,
        src,
        src_len,
        get_str_logits=False,
        get_stack_info=False,
        do_sample=False,
        get_final_answer=False,
    ):
        ### if get_str_logits is true, return the logits for the string
        ### if get_stack_info is true, return the stack operations decoded, as well as their logprobs.
        ### if do_sample is true, sample from the distribution instead of taking the argmax

        batch_size = src.shape[0]
        max_len = src.shape[1]
        # run the first token
        state = self.trafo.encoder.create_state(src.shape[0], max_len, src.device)
        state.step = 0
        curr_mask = None
        curr_closed = np.zeros((batch_size, max_len))
        final_out = [None] * batch_size
        reduced_set = [set() for _ in range(batch_size)]

        if get_str_logits:
            all_str_logits = []

        if get_stack_info:
            all_stack_info = [[]] * batch_size
            all_stack_logprob = []

        import pdb; pdb.set_trace();
        for i in range(max_len):
            running = torch.tensor([i < l for l in src_len]).to(src.device)
            if curr_mask is None:
                curr_mask = ~running.unsqueeze(1)
            else:
                curr_mask = torch.cat([curr_mask, ~running.unsqueeze(1)], dim=1)
            next_tgt = torch.cat(
                [
                    self.pos_embed(self.input_embed(inp[i : i + 1]).unsqueeze(1), i)
                    for inp in src
                ]
            )
            output, attachment_logits = self.trafo.encoder.one_step_forward(
                state,
                next_tgt,
                src_length_mask=curr_mask,
                stack_tape=torch.tensor(curr_closed).unsqueeze(1).to(src.device),
            )

            select_logits_curr = attachment_logits[0].cpu().numpy()
            ### set things that have been reduced to -inf i.e. they cannot be chosen again
            for bs, logit in enumerate(select_logits_curr):
                for elem in reduced_set[bs]:
                    logit[0][elem] = float("-inf")
                # not the final token and not the first token, then cannot reduce with the first token
                if i != src_len[bs] - 1 and i != 0:
                    logit[0][0] = float("-inf")

            if do_sample:
                ### sample from the distribution instead of taking the argmax
                ### convert logits to softmax
                select_logits_curr = torch.tensor(select_logits_curr).to(src.device)
                select_logits_curr = F.softmax(select_logits_curr, dim=-1)[:, 0, :]
                ### sample from the distribution
                attachment_decisions = [
                    torch.multinomial(probs, 1).cpu().numpy()
                    for probs in select_logits_curr
                ]
            else:
                attachment_decisions = select_logits_curr.argmax(axis=-1)

            if get_final_answer:
                ### do not hardcode because this is a math expression
                attachment_decisions = [
                    p[0].item() for bs, p in enumerate(attachment_decisions)
                ]
            else:
                ### hardcode the stack preds to be 0 if the token is the last token in the sequence
                attachment_decisions = [
                    0 if i == src_len[bs] - 1 else p[0].item()
                    for bs, p in enumerate(attachment_decisions)
                ]

            if get_stack_info:
                ### take log softmax of logits_curr and then take the log prob of the chosen stack preds
                ### noting that the stack preds are hard coded to be 0 if the token is the last token in the sequence.
                logits_curr_t = attachment_logits[0]
                logits_curr_t = F.log_softmax(logits_curr_t, dim=-1)
                preds_t = torch.tensor(attachment_decisions).to(src.device)
                log_probs = torch.gather(
                    logits_curr_t, 2, preds_t.unsqueeze(-1).unsqueeze(-1)
                ).squeeze(2)

                all_stack_logprob.append(log_probs)
                all_stack_info = [
                    all_stack_info[bs] + [attachment_decisions[bs]]
                    for bs in range(batch_size)
                ]

            curr_closed, reduced_set = self._apply_reduce(
                curr_closed, attachment_decisions, i, reduced_set
            )
            preds = self.output_map(output[:, 0])
            for idx, _ in enumerate(preds):
                ### if this is the final token
                if i == src_len[idx] - 1:
                    final_out[idx] = preds[idx]

            if get_str_logits:
                all_str_logits.append(preds)

        ret = (final_out,)
        if get_str_logits:
            ret += (torch.stack(all_str_logits),)

        if get_stack_info:
            stack_logprobs = torch.stack(all_stack_logprob)[:, :, 0].transpose(0, 1)
            # take the sum along the second axis but set all logprobs after src_len to 0
            logprobs = [
                stack_logprob[: src_len[i]]
                for i, stack_logprob in enumerate(stack_logprobs.cpu().tolist())
            ]
            ret += (all_stack_info, logprobs)
        return ret

    def run_greedy(
        self, src: torch.Tensor, src_len: torch.Tensor, max_len: int
    ) -> TransformerResult:
        batch_size = src.shape[0]
        n_steps = src.shape[1]

        src = self.pos_embed(self.input_embed(src), 0)
        in_len_mask = self.generate_len_mask(src.shape[1], src_len)
        processed = self.trafo(
            src, tgt=None, src_length_mask=in_len_mask, get_all_layers=True
        )
        running = torch.ones([batch_size], dtype=torch.bool, device=src.device)

        last_embeddings = self.output_map(
            torch.cat(
                [curr[l - 1].unsqueeze(0) for curr, l in zip(processed[-1], src_len)]
            )
        )

        pred_words = torch.argmax(last_embeddings, -1)
        running &= pred_words != self.encoder_eos
        out_len = torch.zeros_like(running, dtype=torch.long)
        out_len[running] = 1
        next_tgt = torch.cat(
            [
                self.pos_embed(
                    self.input_embed(pred_words[idx : idx + 1]).unsqueeze(1), slen
                )
                for idx, slen in enumerate(src_len)
            ]
        )

        all_outputs = [last_embeddings.unsqueeze(1)]
        state = self.trafo.encoder.create_state(
            src.shape[0], n_steps + max_len, src.device
        )

        for idx in range(len(processed) - 1):
            state.state[idx][:, :n_steps] = processed[idx]

        state.step = n_steps
        # pos masking not implemented because it is not needed!
        curr_mask = in_len_mask
        all_attachment_logits = []
        for i in range(max_len):
            curr_mask = torch.cat([curr_mask, ~running.unsqueeze(1)], dim=1)
            ### state is the set of hidden states corresponding to the prefix so far.
            ### next_tgt is the token emitted in the previous turn.
            output, attachment_logits = self.trafo.encoder.one_step_forward(
                state, next_tgt, src_length_mask=curr_mask
            )

            if len(attachment_logits) > 0:
                _, _, steps = attachment_logits[0].shape
                if steps < src.shape[1] + max_len:
                    all_attachment_logits.append(
                        torch.cat(
                            [
                                attachment_logits[0],
                                float("-inf")
                                * torch.ones(
                                    (len(src), 1, src.shape[1] + max_len - steps)
                                ).to(src.device),
                            ],
                            dim=-1,
                        )
                    )
                else:
                    all_attachment_logits.append(attachment_logits[0])
            output = self.output_map(output)
            all_outputs.append(output)
            out_token = torch.argmax(output[:, -1], -1)
            running &= out_token != self.encoder_eos
            out_len[running] += 1
            next_tgt = torch.cat(
                [
                    self.pos_embed(
                        self.input_embed(out_token[idx : idx + 1]).unsqueeze(1),
                        i + 1 + slen,
                    )
                    for idx, slen in enumerate(src_len)
                ]
            )

        if len(all_attachment_logits) > 0:
            all_attachment_logits = torch.cat(all_attachment_logits, dim=1)
        return (
            TransformerResult.create(torch.cat(all_outputs, 1), out_len),
            all_attachment_logits,
        )

    def forward(
        self, src: torch.Tensor, src_len: torch.Tensor, stack_tape=None
    ) -> TransformerResult:
        """
        Run transformer encoder-decoder on some input/output pair

        :param src: source features. Shape: [N, S, D], where S in the in sequence length, N is the batch size
        :param src_len: length of source sequences. Shape: [N], N is the batch size
        :return: prediction of the target tensor. Shape [N, T, C_out]
        """
        src = self.pos_embed(self.input_embed(src), 0)
        if self.embedding_dropout is not None:
            src = self.embedding_dropout(src)
        return self.run_teacher_forcing(src, src_len, stack_tape=stack_tape)
