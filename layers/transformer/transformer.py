import torch
import torch.nn
import torch.nn.functional as F
from .multi_head_relative_pos_attention import PushdownAttention, AttentionMask
from typing import Optional, Callable, Dict
from dataclasses import dataclass

# This file is based on PyTorch's internal implementation

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation: ActivationFunction = F.relu,
        use_stack_tape=False,
        rec_layer_args=None,
        layer_idx=None,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        print(layer_idx)
        self.use_stack_tape = use_stack_tape
        self.use_stack_tape = (
            self.use_stack_tape and layer_idx in rec_layer_args["rec_layers"]
        )
        if self.use_stack_tape:
            print("using pushdown-layer on layer", layer_idx)
            proj_size = d_model // nhead
            depth_embed_vocab_size = rec_layer_args.get("stack_type_vocab", 2)
            depth_embed = torch.nn.Embedding(depth_embed_vocab_size, proj_size)
            self.self_attn = PushdownAttention(
                d_model,
                nhead,
                depth_embed=depth_embed,
                dropout=dropout,
                global_pos_bias=False,
                global_content_bias=False,
                layer_idx=layer_idx,
                rec_layer_args=rec_layer_args,
            )
        else:
            # with no depth embeddings, pushdown attention blocks are standard MHA blocks
            self.self_attn = PushdownAttention(
                d_model,
                nhead,
                depth_embed=None,
                dropout=dropout,
                global_pos_bias=False,
                global_content_bias=False,
                layer_idx=layer_idx,
                rec_layer_args=rec_layer_args,
            )

        dim_feedforward = rec_layer_args.get("dim_feedforward", dim_feedforward)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        full_target=None,
        **kwargs,
    ) -> torch.Tensor:
        pos_offset = kwargs["pos_offset"] if "pos_offset" in kwargs else 0
        if self.use_stack_tape:
            stack_tape = kwargs["stack_tape"]
            if "get_attn_scores" in kwargs:
                src2, attachment_logits, weights = self.self_attn(
                    src,
                    src,
                    AttentionMask(mask, pos_mask),
                    stack_tape=stack_tape,
                    need_weights=True,
                    pos_offset=pos_offset,
                )
            else:
                src2, attachment_logits = self.self_attn(
                    src,
                    src if full_target is None else full_target,
                    AttentionMask(mask, pos_mask),
                    stack_tape=stack_tape,
                    pos_offset=pos_offset,
                )
        else:
            if "get_attn_scores" in kwargs:
                src2, attachment_logits, weights = self.self_attn(
                    src,
                    src,
                    AttentionMask(mask, pos_mask),
                    need_weights=True,
                    pos_offset=pos_offset,
                )
            else:
                src2, attachment_logits = self.self_attn(
                    src,
                    src if full_target is None else full_target,
                    AttentionMask(mask, pos_mask),
                    pos_offset=pos_offset,
                )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if "get_attn_scores" in kwargs:
            return src, weights
        else:
            return src, attachment_logits

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(
            self.linear1.weight,
            gain=torch.nn.init.calculate_gain("relu")
            if self.activation is F.relu
            else 1.0,
        )
        torch.nn.init.xavier_uniform_(self.linear2.weight)


class TransformerEncoder(torch.nn.Module):
    @dataclass
    class State:
        step: int
        state: Dict[int, torch.Tensor]

    def __init__(self, layer, n_layers: int, *args, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [layer(*args, layer_idx=lidx, **kwargs) for lidx in range(n_layers)]
        )

        self.n_layers = n_layers
        self.d_model = self.layers[0].d_model

    def create_state(
        self, batch_size: int, max_length: int, device: torch.device
    ) -> State:
        # len(self.layers) +1 to store the final (pre-softmax) hidden states!
        return self.State(
            0,
            {
                i: torch.zeros([batch_size, max_length, self.d_model], device=device)
                for i in range(len(self.layers) + 1)
            },
        )

    def attn_matrices(self, data: torch.Tensor, src_length_mask, pos_mask, **kwargs):
        attn_matrices = []
        for idx, l in enumerate(self.layers):
            _, mat = l(
                data,
                mask=src_length_mask,
                get_attn_scores=True,
                pos_mask=pos_mask,
                **kwargs,
            )
            attn_matrices.append(mat)
        return attn_matrices

    def forward(self, data: torch.Tensor, *args, **kwargs):
        if "src_length_mask" in kwargs:
            mask = kwargs["src_length_mask"]
        elif len(args) > 0:
            mask = args[0]
        else:
            mask = None

        if "layer_id" in kwargs:
            layer_id = kwargs["layer_id"]
        else:
            layer_id = -1

        if "get_all_layers" in kwargs:
            all_data = [data]

        attachment_logits_all = []
        for idx, l in enumerate(self.layers):
            if type(mask) == list:
                mask_curr = mask[idx]
            else:
                mask_curr = mask
            data, attachment_logits = l(data, mask=mask_curr, **kwargs)
            if attachment_logits is not None:
                attachment_logits_all.append(attachment_logits)
            if layer_id == idx:
                break
            if "get_all_layers" in kwargs:
                all_data.append(data)
        if "get_all_layers" in kwargs:
            return all_data
        else:
            return data, attachment_logits_all

    def one_step_forward(self, state: State, data: torch.Tensor, *args, **kwargs):
        assert (
            data.shape[1] == 1
        ), f"For one-step forward should have one timesteps, but shape is {data.shape}"
        assert state.step < state.state[0].shape[1]
        if "src_length_mask" in kwargs:
            mask = kwargs["src_length_mask"]
        else:
            mask = None

        attachment_logits_all = []
        for i, l in enumerate(self.layers):
            state.state[i][:, state.step : state.step + 1] = data
            data, attachment_logits = l(
                data,
                mask=mask,
                full_target=state.state[i][:, : state.step + 1],
                pos_offset=state.step,
                **kwargs,  # state.step,
            )
            if attachment_logits is not None:
                attachment_logits_all.append(attachment_logits)

        ### store the final hidden states after processing thru the entire transformer

        state.state[len(self.layers)][:, state.step : state.step + 1] = data
        state.step += 1
        return data, attachment_logits_all


def TransformerEncoderWithLayer(layer=TransformerEncoderLayer):
    return lambda *args, **kwargs: TransformerEncoder(layer, *args, **kwargs)



class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: ActivationFunction = F.relu,
        use_stack_tape=False,
        encoder_layer=TransformerEncoderWithLayer(),
        is_null_encoder=False,
        **kwargs,
    ):
        super().__init__()
        self.use_stack_tape = use_stack_tape

        rec_layer_args = kwargs["recursive_layer_args"]

        self.rec_layer_config = rec_layer_args

        if is_null_encoder:
            self.encoder = lambda src, src_length_mask: src
            self.num_encoder_layers = 0
        else:
            self.encoder = encoder_layer(
                num_encoder_layers,
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                use_stack_tape=use_stack_tape,
                rec_layer_args=rec_layer_args,
            )
            self.num_encoder_layers = num_encoder_layers

    def get_hidden_states(self, src, src_length_mask=None, layer_id=-1, is_lm=False):
        if is_lm:
            if type(src_length_mask) == list:
                pos_mask = self.generate_square_subsequent_mask(
                    src_length_mask[0].shape[1], device=src.device
                )
            else:
                pos_mask = self.generate_square_subsequent_mask(
                    src_length_mask.shape[1], device=src.device
                )
            memory = self.encoder(
                src,
                src_length_mask=src_length_mask,
                layer_id=layer_id,
                pos_mask=pos_mask,
            )
        else:
            memory = self.encoder(
                src,
                src_length_mask=src_length_mask,
                layer_id=layer_id,
            )

        return memory

    def get_attn_matrices(self, src, tgt, stack_tape=None, src_length_mask=None):
        if tgt is None:
            pos_mask = self.generate_square_subsequent_mask(
                src_length_mask.shape[1], device=src.device
            )
        else:
            pos_mask = None
        attn_matrices = self.encoder.attn_matrices(
            src,
            stack_tape=stack_tape,
            src_length_mask=src_length_mask,
            pos_mask=pos_mask,
        )
        return attn_matrices

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_length_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        assert tgt is None
        # run as a left to right language model
        pos_mask = self.generate_square_subsequent_mask(
            src_length_mask.shape[1], device=src.device
        )
        return self.encoder(
            src, src_length_mask=src_length_mask, pos_mask=pos_mask, **kwargs
        )

    def generate_square_subsequent_mask(
        self, sz: int, device: torch.device
    ) -> torch.Tensor:
        return torch.triu(
            torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1
        )
