from typing import Callable, Optional
import torch
from layers import Transformer
from models.transformer_lm import TransformerResult
from .transformer_lm import TransformerLM
from layers import Transformer, TiedEmbedding, PositionalEncoding
from typing import Callable, Optional
from layers.transformer.multi_head_relative_pos_attention import AttentionMask
from layers.transformer.multi_head_relative_pos_attention import PushdownAttention

import numpy as np
import torch.nn.functional as F


class StackPredictor(PushdownAttention):
    def __init__(
        self,
        state_size: int,
        n_heads: int,
        dropout: float = 0.1,
        global_pos_bias: bool = True,
        global_content_bias: bool = True,
        input_size: Optional[int] = None,
        rec_layer_args=None,
    ):
        if rec_layer_args:
            depth_embed_vocab_size = rec_layer_args.get("stack_type_vocab", 2)
        else:
            depth_embed_vocab_size = 2
        projection_size = state_size // n_heads

        self.additive_composition = rec_layer_args.get("additive_composition", False)

        if self.additive_composition:
            # we will add the depth embed to the key
            depth_embed = torch.nn.Embedding(depth_embed_vocab_size, 2 * state_size)
        else:
            # we will use a small NN to compose the key and depth embed
            depth_embed = torch.nn.Embedding(depth_embed_vocab_size, projection_size)
        super().__init__(
            state_size,
            n_heads,
            dropout,
            global_pos_bias,
            global_content_bias,
            input_size,
            depth_embed=depth_embed,
            only_attachment_decisions=True,
            layer_idx=rec_layer_args["stack_pred_layer"],
            rec_layer_args=rec_layer_args,
        )
        self.register_buffer("int_seq", torch.arange(5000, dtype=torch.long))

        self.q_next_word_mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * state_size, 2 * state_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(2 * state_size, 2 * state_size),
        )

        self.k_next_word_mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * state_size, 2 * state_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(2 * state_size, 2 * state_size),
        )

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[:max_len] >= len.unsqueeze(-1)

    def forward(
        self,
        attend_to,
        curr_state,
        next_word,
        stack_tape,
        mask,
        pos_offset,
        take_step_mode=False,
    ):
        ## attend_to has shape: [n_batch, n_src, vec_dim]
        ## curr_state has shape: [n_batch, n_dest, vec_dim]
        # stack_tape has shape: [n_batch, n_dest, n_src]
        # next_word has shape: [n_batch, n_dest, vec_dim]
        n_src = attend_to.shape[1]
        n_dest = curr_state.shape[1]
        batch_size, in_len = attend_to.shape[0:2]

        q_stack = self.data_to_q_prime(curr_state)

        ### combine q_stack with next_w using q_next_word_mlp, to get a *contextual representation* of the next word
        q_stack = self.q_next_word_mlp(
            torch.cat(
                [
                    q_stack,
                    next_word,
                ],
                dim=-1,
            )
        )
        # q_stack has shape [n_batch, n_dest, vec_dim]

        next_word_key = self.k_next_word_mlp(
            torch.cat(
                [
                    q_stack,
                    next_word,
                ],
                dim=-1,
            )
        ).unsqueeze(2)
        # next_word_key has shape [n_batch, n_dest, 1, vec_dim]
        if self.key_and_stack_info_composer is not None:
            stack_info = stack_tape[:, :n_dest, :n_src]
            # depth_embed_biases has shape [n_batch, n_dest, n_src, vec_dim]
            depth_embed_biases = self.depth_embed(stack_info.int())

            k_prime = (
                self.data_to_k_prime(attend_to).unsqueeze(1).repeat(1, n_dest, 1, 1)
            )
            ## k_stack_composed has shape [n_batch, n_dest, n_src, vec_dim]
            if self.additive_composition:
                k_stack_composed = k_prime + depth_embed_biases
            else:
                k_stack_composed = self.key_and_stack_info_composer(
                    torch.cat([k_prime, depth_embed_biases], dim=-1)
                )
            logits = torch.matmul(
                q_stack.unsqueeze(2), k_stack_composed.transpose(2, 3)
            ).squeeze(2)
        else:
            # [n_batch, n_src, vec_dim]
            k_prime = self.data_to_k_prime(attend_to)
            logits = torch.matmul(q_stack, k_prime.transpose(1, 2))

        ## logits has shape [n_batch, n_dest, n_src], this more or less decides which elem to reduce next word with.
        logit_self = torch.matmul(
            q_stack.unsqueeze(2), next_word_key.transpose(2, 3)
        ).squeeze(2)
        ### however if we want to reduce next_word with itself (i.e. no reduce!), we need to add a logit_self.

        ### logit_self has shape [n_batch, n_dest, 1]

        if take_step_mode:
            logits_all = torch.cat([logits, logit_self], dim=-1)
        else:
            ### insert logit_self into the off-diagonal positions of logits. i.e. [n_batch, n_dest, n_src] -> [n_batch, n_dest, n_src + 1]
            logits_l = torch.cat(
                [
                    logits,
                    torch.zeros(
                        logits.shape[0],
                        logits.shape[1],
                        1,
                        device=logits.device,
                    ),
                ],
                dim=-1,
            )
            logits_all = logits_l.scatter(
                2,
                (1 + torch.arange(n_dest))
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(batch_size, 1, 1)
                .to(logits_l.device),
                logit_self,
            )

        n_batch = q_stack.shape[0]
        n_out_steps = q_stack.shape[1]

        ### add position information here.
        k_pos_stack = self.pos_to_pq_stack(
            self.get_pos(in_len + 1, pos_offset)
        ).transpose(0, 1)

        # we do this to get the semantics that j+1th position in row j is the "self" position i.e. relative position 0.
        dummy = torch.zeros_like(q_stack[:, :1, :]).to(q_stack.device)
        q_stack_dummy = torch.cat([dummy, q_stack], dim=1)
        pos_info = torch.matmul(
            q_stack_dummy, k_pos_stack
        )  # [n_batch, n_out, n_in * 2 - 1]
        pos_info = self._shift(pos_info)
        attachment_logits = logits_all + pos_info[:, 1:, :]

        if mask.position_mask is not None:
            str_lens = torch.sum(~mask.src_length_mask, dim=-1) + 1
            ### create a mask so any position longer than str_lens is masked out
            pos_mask_2 = self.generate_len_mask(
                mask.src_length_mask.shape[1] + 1, str_lens
            )
            attachment_logits = attachment_logits.masked_fill(
                pos_mask_2.unsqueeze(1), float("-inf")
            )
            attachment_logits = attachment_logits.masked_fill(
                mask.src_length_mask.unsqueeze(2), float("-inf")
            )
            attachment_logits = attachment_logits.masked_fill(
                mask.position_mask.unsqueeze(0), float("-inf")
            )
        return attachment_logits


class PushdownLM(TransformerLM):
    def __init__(
        self,
        n_input_tokens: int,
        state_size: int = 512,
        ff_multiplier: float = 1,
        max_len: int = 5000,
        embedding_dropout: float = -1,
        output_dropout: float = -1,
        transformer=Transformer,
        tied_embedding: bool = False,
        pos_embedding=None,
        encoder_sos: bool = True,
        embedding_init: str = "pytorch",
        scale_mode: str = "none",
        **kwargs
    ):
        super().__init__(
            n_input_tokens,
            state_size,
            ff_multiplier,
            max_len,
            embedding_dropout,
            output_dropout,
            transformer,
            tied_embedding,
            pos_embedding,
            encoder_sos,
            embedding_init,
            scale_mode,
            **kwargs
        )

        self.synchronous = True
        self.n_heads = ff_multiplier
        self.attachment_head = StackPredictor(
            self.state_size,
            self.n_heads,
            global_content_bias=False,
            rec_layer_args=kwargs.get("recursive_layer_args", None),
        )

    def run_teacher_forcing(
        self, src, next_steps, src_len, stack_tape
    ) -> TransformerResult:
        in_len_mask = self.generate_len_mask(src.shape[1], src_len)
        res, _ = self.trafo(
            src, tgt=None, src_length_mask=in_len_mask, stack_tape=stack_tape
        )

        pos_mask = self.trafo.generate_square_subsequent_mask(
            in_len_mask.shape[1] + 1, device=src.device
        )

        ret_dict = {"output": TransformerResult.create(self.output_map(res), src_len)}
        mask = AttentionMask(in_len_mask, pos_mask[1:])
        attachment_logits = self.attachment_head(
            res, res, self.input_embedding(next_steps), stack_tape, mask, pos_offset=0
        )
        if self.output_dropout is not None:
            res = self.output_dropout(res)

        ret_dict["attachment_logits"] = [attachment_logits]
        return ret_dict

    def _take_step_with_stack_tape(
        self, src, trafo_state, stack_state, list_of_reduced, step, return_preds=False
    ):
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
        output, _ = self.trafo.encoder.one_step_forward(
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
        if return_preds:
            return preds

        pred_logprob = F.log_softmax(preds, dim=-1)  # bs x vocab_size
        ## get log prob of targets
        str_logprobs = torch.gather(pred_logprob, 1, targets.unsqueeze(1)).squeeze(1)

        ### this is a synchronous LM, so we use targets to make stack predictions
        final_layer = max([key for key in trafo_state.state])
        attend_to = trafo_state.state[final_layer][
            :, : trafo_state.step
        ]  # bs x (1+step) x state_size
        curr_state = output  ### bs x 1 x state_size

        attachment_logits = self.attachment_head(
            attend_to,
            curr_state,
            self.input_embedding(targets).unsqueeze(1),
            torch.tensor(stack_state).unsqueeze(1).to(src.device),
            AttentionMask(None, None),
            pos_offset=step,
            take_step_mode=True,
        )

        ### logits_curr_t: bs x (step + 1)
        logits_curr_t = attachment_logits[:, 0, :]
        # print(logits_curr_t)
        logprobs_curr = F.log_softmax(logits_curr_t, dim=-1)

        ### logprobs corresponding to things that have been reduced should be -inf
        for bs, reduced_set in enumerate(list_of_reduced):
            for elem in reduced_set:
                logprobs_curr[bs][elem] = float("-inf")

        return logprobs_curr, str_logprobs, trafo_state

    def get_attention_matrices(self, src, src_len, stack_tape):
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

    def forward(self, src, next_steps, src_len, stack_tape) -> TransformerResult:
        src = self.pos_embed(self.input_embed(src), 0)
        if self.embedding_dropout is not None:
            src = self.embedding_dropout(src)
        return self.run_teacher_forcing(src, next_steps, src_len, stack_tape)
