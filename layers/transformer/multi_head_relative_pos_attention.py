import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional
import layers
import math
from dataclasses import dataclass
from typing import Optional, Callable, List, Union, Tuple

@dataclass
class AttentionMask:
    src_length_mask: Optional[torch.Tensor]
    position_mask: Optional[torch.Tensor]

class MultiHeadAttentionBase(torch.nn.Module):
    def __init__(self, state_size: int, n_heads: int, dropout: float = 0.1):
        assert state_size % n_heads == 0
        super().__init__()
        self.state_size = state_size
        self.projection_size = state_size // n_heads
        self.n_heads = n_heads
        self.scale = 1.0 / math.sqrt(self.projection_size)

        self.dropout = torch.nn.Dropout(dropout)
        self.multi_head_merge = torch.nn.Linear(
            n_heads * self.projection_size, state_size, bias=False
        )

    def _masked_softmax(
        self, logits: torch.Tensor, mask: Optional[AttentionMask]
    ) -> torch.Tensor:
        if mask is None or (
            mask.src_length_mask is None and mask.position_mask is None
        ):
            return F.softmax(logits, -1)

        # Output shape: [n_batch * n_heads, n_time_dest, n_time_src]
        bb, n_time_dest, n_time_src = logits.shape

        logits = logits.view(bb // self.n_heads, self.n_heads, n_time_dest, n_time_src)
        if mask.position_mask is not None:
            logits = logits.masked_fill(
                mask.position_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        if mask.src_length_mask is not None:
            logits = logits.masked_fill(
                mask.src_length_mask.unsqueeze(1).unsqueeze(1), float("-inf")
            )

        logits = F.softmax(logits, -1)
        return logits.view(bb, n_time_dest, n_time_src)

    def _attention_read(
        self, mask: Optional[AttentionMask], logits: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # logits: [n_batch * n_heads, n_out, n_in]
        # v: [n_nbatch * n_heads, n_in]
        # Output data shape [n_batch * n_heads, n_time_dest, data_size]
        # Out attention score shape: [n_batch, n_heads, n_time_dest, n_time_src]
        scores = self._masked_softmax(logits * self.scale, mask)
        scores = self.dropout(scores)

        ### add optional
        return torch.bmm(scores, v), scores.view(-1, self.n_heads, *scores.shape[1:])

    def merged_attention(
        self,
        n_batch: int,
        n_out_steps: int,
        *args,
        need_weights: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        data, scores = self._attention(*args, **kwargs)

        data = (
            data.view(n_batch, self.n_heads, n_out_steps, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(n_batch, n_out_steps, -1)
        )

        return self.multi_head_merge(data), scores

    def transform_data(
        self,
        input: torch.Tensor,
        proj: Callable[[torch.Tensor], torch.Tensor],
        n_projs: int,
    ) -> List[torch.Tensor]:
        # Input shape: [n_batch, n_steps, n_channels]
        # Output: Tuple of n_projs tensors of dimension: [n_batch * n_heads, n_steps, projection_size]
        n_batch, n_steps, _ = input.shape
        transformed = (
            proj(input)
            .view(n_batch, n_steps, self.n_heads, n_projs, self.projection_size)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(n_batch * self.n_heads, n_steps, n_projs, self.projection_size)
        )
        return transformed.unbind(dim=2)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.multi_head_merge.weight)



class RelativeAttentionBase(MultiHeadAttentionBase):
    def __init__(self, state_size: int, n_heads: int, dropout: float):
        super().__init__(state_size, n_heads, dropout=dropout)

    def _shift(self, posmat: torch.Tensor) -> torch.Tensor:
        # Slice out a matrix diagonally. Each successive row is sliced one position to the left compared.
        # shape: [n_batch, n_head, n_out, n_in * 2 - 1]
        # return: [n_batch, n_head, n_out, n_in]
        p = F.pad(posmat, (0, 1, 0, 1)).flatten(
            -2
        )  # [n_batch, n_head, (n_out + 1) * n_in * 2]
        p = p.narrow(
            -1, posmat.shape[-1] // 2, posmat.shape[-1] * posmat.shape[-2]
        ).view_as(posmat)

        return p.narrow(-1, 0, (posmat.shape[-1] + 1) // 2)

    def _attention(
        self,
        mask: Optional[torch.Tensor],
        q_content: torch.Tensor,
        k_content: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        v: torch.Tensor,
        stack_tape=None,
    ) -> torch.Tensor:
        # shape of q_content, q_pos, k_pos: [n_batch * n_heads, n_steps, data_size]
        # k_pos: [n_heads, n_in * 2 - 1, data_size]
        # Output shape [n_batch * n_heads, n_out, data_size]
        n_batch = q_content.shape[0] // self.n_heads
        n_out_steps = q_content.shape[1]

        # content-pos addressing.
        pos = torch.matmul(
            q_pos.view(n_batch, self.n_heads, n_out_steps, -1), k_pos.transpose(-1, -2)
        )  # [n_batch, n_head, n_out, n_in * 2 - 1]
        pos = self._shift(pos).flatten(0, 1)

        if self.use_stack_info:
            bb, n_src, vec_dim = k_content.shape
            n_dest = q_content.shape[1]
            # content-content addressing
            content = torch.bmm(q_content, k_content.transpose(1, 2))
            # (n_batch * heads, n_dest, vec_dim, 1)
            q_content_curr = q_content.unsqueeze(-1)

            # (n_batch, n_dest, n_src) at decoding time, need just a bs x 1 x src matrix
            stack_info = stack_tape[:, :n_dest, :n_src]
            depth_biases = (
                self.depth_embed(stack_info.int())
                .unsqueeze(1)
                .repeat(1, self.n_heads, 1, 1, 1)
            )

            stack_scores = (
                torch.bmm(
                    depth_biases.view(-1, n_src, vec_dim).float(),
                    q_content_curr.view(-1, vec_dim, 1).float(),
                )
                .squeeze(-1)
                .view(bb, n_dest, n_src)
            )

            overall_scores = content + pos + stack_scores
        else:
            content = torch.bmm(q_content, k_content.transpose(1, 2))
            overall_scores = content + pos

        # Logits shape: [n_batch * n_heads, n_out, n_in]
        return self._attention_read(mask, overall_scores, v)

    def _get_pos_subset(
        self, pos_encoding: torch.Tensor, length: int, offset: int
    ) -> torch.Tensor:
        l_slice = 2 * length - 1
        assert pos_encoding.shape[0] > l_slice
        return pos_encoding.narrow(
            0, pos_encoding.shape[0] // 2 - length + 1 - offset, 2 * length - 1
        )


class PushdownAttention(RelativeAttentionBase):
    def __init__(
        self,
        state_size: int,
        n_heads: int,
        dropout: float = 0.0,
        global_pos_bias: bool = True,
        global_content_bias: bool = True,
        input_size: Optional[int] = None,
        depth_embed=None,
        layer_idx=-1,
        rec_layer_args=None,
        only_attachment_decisions=False,
    ):
        super().__init__(state_size, n_heads, dropout)

        self.data_to_q = torch.nn.Linear(
            state_size if input_size is None else input_size,
            n_heads * self.projection_size,
            bias=False,
        )
        if not only_attachment_decisions:
            self.data_to_kv = torch.nn.Linear(
                state_size, 2 * n_heads * self.projection_size, bias=False
            )
            self.pos_to_pq = torch.nn.Linear(
                state_size, self.n_heads * self.projection_size, bias=False
            )

        self.global_content_bias = (
            torch.nn.Parameter(torch.zeros([n_heads, self.projection_size]))
            if global_content_bias
            else None
        )
        self.global_pos_bias = (
            torch.nn.Parameter(torch.zeros([n_heads, self.projection_size]))
            if global_pos_bias
            else None
        )
        self.layer_idx = layer_idx

        if rec_layer_args is not None:
            ### if we are a synchronous LM, we don't want to make stack predictions using attention layers, but instead have a separate module
            self.compose_keys_and_stack_info = rec_layer_args.get(
                "compose_keys_and_stack_info", False
            )
        else:
            self.compose_keys_and_stack_info = False

        self.make_attachment_decisions = only_attachment_decisions
        if only_attachment_decisions:
            print("making attachment predictions at this layer")
            self.pos_to_pq_stack = torch.nn.Linear(
                state_size, 2 * state_size, bias=False
            )
            self.data_to_k_prime = torch.nn.Linear(
                state_size, 2 * state_size, bias=False
            )
            self.data_to_q_prime = torch.nn.Linear(
                state_size, 2 * state_size, bias=False
            )


        self.depth_embed = depth_embed
        self.as_biases = False
        self.use_stack_info = False
        if depth_embed is not None:
            print("using stack info at this layer")
            self.use_stack_info = True
            if type(self.depth_embed) == torch.nn.modules.sparse.Embedding:
                self.as_biases = True
            if self.compose_keys_and_stack_info and self.make_attachment_decisions:
                ### a deep net combines the stack info and the keys
                mlp_in_dim = 2 * state_size + self.projection_size
                mlp_out_dim = 2 * state_size
                self.key_and_stack_info_composer = torch.nn.Sequential(
                    torch.nn.Linear(mlp_in_dim, mlp_in_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(mlp_in_dim, mlp_out_dim),
                )
            else:
                self.key_and_stack_info_composer = None

        self.register_buffer("pos_encoding", self._create_buffer(2048))

    def _create_buffer(self, max_len: int):
        return layers.sinusoidal_pos_embedding(
            self.state_size,
            2 * max_len - 1,
            -max_len + 1,
            device=self.data_to_q.weight.device,
        )

    def get_pos(self, l: int, offset: int) -> torch.Tensor:
        if self.pos_encoding.shape[0] < 2 * (l + offset) - 1:
            self.pos_encoding = self._create_buffer(
                int(2 ** math.ceil(math.log2(2 * (l + offset) - 1)))
            )

        return self._get_pos_subset(self.pos_encoding, l, offset)

    def add_head_specific_bias(
        self, data: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # data [batch * n_heads, len, c]
        # bias [n_heads, c]
        return (
            (
                data.view(-1, bias.shape[0], *data.shape[1:])
                + bias.unsqueeze(1).type_as(data)
            ).view_as(data)
            if bias is not None
            else data
        )


    def forward(
        self,
        curr_state: torch.Tensor,
        attend_to: torch.Tensor,
        mask: Optional[AttentionMask],
        pos_offset: int = 0,
        need_weights: bool = False,
        stack_tape=None,
    ):
        # curr_state: [batch_size, out_len, c]
        # attend_to: [batch_size, in_len, c]
        batch_size, in_len = attend_to.shape[0:2]
        out_len = curr_state.shape[1]
        k_content, v = self.transform_data(attend_to, self.data_to_kv, 2)
        (q,) = self.transform_data(curr_state, self.data_to_q, 1)

        k_pos = (
            self.pos_to_pq(self.get_pos(in_len, pos_offset))
            .view(-1, self.n_heads, self.projection_size)
            .transpose(0, 1)
        )  # n_heads, 2*in_len -1 , projection_size

        q_content = self.add_head_specific_bias(q, self.global_content_bias)
        q_pos = self.add_head_specific_bias(q, self.global_pos_bias)
        data, scores = self.merged_attention(
            batch_size,
            out_len,
            mask,
            q_content,
            k_content,
            q_pos,
            k_pos,
            v,
            need_weights=need_weights,
            stack_tape=stack_tape,
        )

        if need_weights:
            # Calculate the mean over the heads
            return data, None, scores.mean(1)
        else:
            return data, None

    def reset_parameters(self):
        super().reset_parameters()

        torch.nn.init.xavier_uniform_(self.data_to_q.weight)
        torch.nn.init.xavier_uniform_(self.pos_to_pq.weight)
        torch.nn.init.xavier_uniform_(
            self.data_to_kv.weight[: self.data_to_kv.weight.shape[0] // 2]
        )
        torch.nn.init.xavier_uniform_(
            self.data_to_kv.weight[self.data_to_kv.weight.shape[0] // 2 :]
        )

        if self.global_content_bias is not None:
            self.global_content_bias.fill_(0)

        if self.global_pos_bias is not None:
            self.global_pos_bias.fill_(0)
