import torch
import torch.nn
from typing import Dict, Tuple
import torch.nn.functional as F
import layers
import models

import torch
from typing import Dict, Any, List, Optional
import torch
from dataclasses import dataclass
from typing import List, Optional


class Result:
    outputs: torch.Tensor
    loss: torch.Tensor

    batch_dim = 0

    def plot(self) -> Dict[str, Any]:
        return {}

    @property
    def batch_size(self) -> int:
        return self.outputs.shape[self.batch_dim]

    @staticmethod
    def merge(l: List, batch_weights: Optional[List[float]] = None):
        if len(l) == 1:
            return l[0]
        batch_weights = batch_weights if batch_weights is not None else [1] * len(l)
        loss = sum([r.loss * w for r, w in zip(l, batch_weights)]) / sum(batch_weights)
        out = torch.stack([r.outputs for r in l], l[0].batch_dim)
        return l[0].__class__(out, loss)

@dataclass
class EncoderDecoderResult(Result):
    outputs: torch.Tensor
    out_lengths: torch.Tensor
    loss: torch.Tensor

    batch_dim = 1

    @staticmethod
    def merge(l: List, batch_weights: Optional[List[float]] = None):
        if len(l) == 1:
            return l[0]
        batch_weights = batch_weights if batch_weights is not None else [1] * len(l)
        loss = sum([r.loss * w for r, w in zip(l, batch_weights)]) / sum(batch_weights)
        out = torch.stack([r.outputs for r in l], l[0].batch_dim)
        lens = torch.stack([r.out_lengths for r in l], 0)
        return l[0].__class__(out, lens, loss)

def add_eos(input: torch.Tensor, lengths: torch.Tensor, eos_id: int):
    input = torch.cat((input, torch.zeros_like(input[0:1])), dim=0)
    input.scatter_(0, lengths.unsqueeze(0).long(), value=eos_id)
    return input


class TransformerLMInterface():
    def __init__(
        self,
        model: torch.nn.Module,
        in_vocab=None,
        label_smoothing: float = 0.0,
    ):
        self.model = model
        self.vocab = in_vocab
        self.label_smoothing = label_smoothing

    def loss(
        self,
        outputs: torch.Tensor,
        ref: torch.Tensor,
        mask: torch.Tensor,
        normalize,
    ) -> torch.Tensor:
        l = layers.cross_entropy(
            outputs, ref, reduction="none", smoothing=self.label_smoothing
        )
        l = l.reshape_as(ref) * mask
        if normalize:
            return l.sum() / mask.sum()
        else:
            # lens = mask.sum(dim=0)
            # ignore_eos_nll = torch.tensor([l[:, i][:_len-1].sum() for i, _len in enumerate(lens)])
            # return ignore_eos_nll.sum()
            return l.sum()


    def decode_outputs(self, outputs) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(self, data: Dict[str, torch.Tensor], normalize=True):
        in_len = data["in_len"].long() + 1

        sos_tensor = torch.ones((1, data["in"].shape[1])) * self.model.encoder_sos
        sos_tensor = sos_tensor.to(data["in"].device)
        inp_data = torch.cat(
            [sos_tensor, data["in"]],
            dim=0,
        ).transpose(0, 1)

        out_data = add_eos(
            data["in"], data["in_len"], self.model.encoder_eos
        ).transpose(0, 1)
        if "stack_tape" in data:
            stack_tape = data["stack_tape"].transpose(0, 1)
        else:
            stack_tape = None
        # inp_data =  bs x seq_len: [SOS] a b c
        # out_data =  bs x seq_len e.g.  a b c [EOS]

        if self.model.synchronous:
            ret_dict = self.model(inp_data, out_data, in_len, stack_tape)
        else:
            ret_dict = self.model(inp_data, in_len, stack_tape)
        if (
            "attachment_logits" in ret_dict
            and "attachment_labels" in data
            and len(ret_dict["attachment_logits"]) > 0
        ):
            attachment_logits = ret_dict["attachment_logits"][0]
            if self.model.synchronous:
                ### the last token to predict is </s> and we don't want a decision for that.
                attachment_labels = data["attachment_labels"].transpose(0, 1)
                ### attachment_labels has shape bs x seq_len. Add a column of zeros to the end
                ### to make it bs x seq_len+1. This will have the affect of ignoring the attachment decision for </s>
                attachment_labels = torch.cat(
                    [
                        attachment_labels,
                        torch.zeros(
                            (attachment_labels.shape[0], 1), dtype=attachment_labels.dtype
                        ).to(attachment_labels.device),
                    ],
                    dim=1,
                )

                ### ignore the last token in the attachment decision. i.e. when processing <eos> and outputing </s>, we don't care about the attachment decision
                attachment_mask = torch.arange(attachment_labels.shape[1]).to(
                    attachment_labels.device
                ) < (in_len - 1).unsqueeze(1)

            else:
                attachment_labels = data["attachment_labels"].transpose(0, 1)
                attachment_mask = torch.arange(attachment_labels.shape[1]).to(
                    attachment_labels.device
                ) < in_len.unsqueeze(1)

            ### attachment_labels is a 2D tensor with real length given by in_len
            ### e.g. if the first row is [0, 1, 2, 3, 0, 0] and in_len is 5, then the last 0 is padding
            ### mask out all padding in attachment_labels with -100

            ### if mask says 0, then we set the label to -100
            attachment_labels[~attachment_mask] = -100
            attachment_loss = self.loss(
                attachment_logits,
                attachment_labels,
                attachment_mask,
                normalize,
            )

            ### accuracy of greedily decoded parse is a bad measure because of incremental nature of parsing
            attachment_acc = (
                ((attachment_logits.argmax(-1) == attachment_labels).float() * attachment_mask)
                .sum()
                .item()
            )
            total_labels = attachment_mask.sum().item()
            attachment_acc = (attachment_acc, total_labels)
        else:
            attachment_loss = 0.0
            attachment_acc = (0.0, 0.0)

        res = ret_dict["output"]
        res.data = res.data.transpose(0, 1)
        len_mask = ~self.model.generate_len_mask(
            inp_data.shape[1], in_len
        ).transpose(0, 1)
        loss = self.loss(res.data, out_data.transpose(0, 1), len_mask, normalize)
        return {
            "lm_out": EncoderDecoderResult(res.data, res.length, loss),
            "attachment_loss": attachment_loss,
            "attachment_acc": attachment_acc,
        }
