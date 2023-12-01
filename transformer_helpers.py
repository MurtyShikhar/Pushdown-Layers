import torch.nn
from layers.transformer import Transformer
from interfaces import TransformerLMInterface
from models.transformer_lm import TransformerLM
from models.pushdown_transformer_lm import PushdownLM


def create_lm(
    t_args,
    in_vocab_size,
    vec_dim,
    n_heads,
    encoder_n_layers,
    ff_multiplier=1,
    use_stack_tape=False,
    recursive_layer_args=None,
) -> torch.nn.Module:
    args = dict(embedding_init="xavier", scale_mode="opennmt")

    # we use relative position embeddings
    args["pos_embedding"] = lambda x, offset: x
    args["dropout"] = t_args.get("dropout", 0.1)
    args["embedding_dropout"] = t_args.get("embedding_dropout", -1.0)
    args["output_dropout"] = t_args.get("output_dropout", -1.0)
    return PushdownLM(
        in_vocab_size,
        vec_dim,
        n_heads,
        num_encoder_layers=encoder_n_layers,
        use_stack_tape=use_stack_tape,
        recursive_layer_args=recursive_layer_args,
        tied_embedding=True,
        **args,
    )



def create_model_interface(
    model,
    in_vocab=None,
    label_smoothing=0.0,
    is_null_encoder=False,
    is_lm=False,
):
    return TransformerLMInterface(
        model,
        in_vocab=in_vocab,
        label_smoothing=label_smoothing,
    )


