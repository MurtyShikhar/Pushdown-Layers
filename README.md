# Pushdown Transformer LMs
Code Repository for our EMNLP '23 paper: "Pushdown Layers: Encoding Recursive Structure in Transformer Language Models".

## Setup Environment
```
conda env create -f environment.yml
conda activate pushdown-lm
```

## Quickstart (I just want to run inference on some sentences)

Download a pre-trained pushdown-LM (16 layers transformer with pushdown self-attention trained on `BLLIP-LG`):
```
# install huggingface-cli if you do not have it
pip install -U "huggingface_hub[cli]" 
```

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="smurty/pushdown-bllip", filename="model.pt", local_dir="bllip-lg");
hf_hub_download(repo_id="smurty/pushdown-bllip", filename="vocab.pkl", local_dir="bllip-lg");
```

We provide a script for running inference on a pushdown-LM in `eval_utils/eval_pushdown_model.py`. This script can be used to score sentences and obtain parses. Given a file `data_utils/sample_sents.txt` containing sentences you want to score and parse via a trained Pushdown-LM, simply run the following:

```
cd eval_utils
python eval_pushdown_model.py settings.model_dir=../bllip-lg settings.eval_mode=beam settings.dataset="../data_utils/sample_sents.txt" settings.shard_id=-1 settings.dir_name=sent_outputs
```


## Get Datasets

### Dyck

We provide the generated dyck data in `data_utils/dyck_data/`, produced using scripts in `data_utils/dyck_helpers.py`

### BLLIP

To get the `BLLIP-LG` datasets we use for training our models, please follow the instructions [here](https://github.com/IBM/transformers-struct-guidance). Once you have `BLLIP-LG` store it inside `data_utils` and run:

```
cd data_utils
python process_bllip.py 
```

This script tokenizes the dataset using the GPT2 tokenizer, and converts every parse tree into unlabeled binarized trees.

### WikiTrees

WIP. Please send Shikhar Murty an email (can be found on the paper link) if you urgently need the WikiTrees dataset.

## Training
Note: This repository uses WandB for logging. Make sure you set your WandB account before training.

### Pushdown LMs on Dyck
To train 6 layer pushdown LMs on the `Dyck` dataset, run:
```
python train_transformers.py train.dataset=dyck train.use_stack_tape=True train.vec_dim=128 train.callback=True train.save_dir=dyck_model-with-stack-acc-mlp-stack-key-modulator +recursive_layer.compose_keys_and_stack_info=True train.max_steps=10000 train.eval_every=500
```

### Pushdown LMs on BLLIP

To train 16 layer pushdown LMs on `BLLIP-LG`, run:

```
python train_transformers.py train.encoder_n_layers=16 train.dataset=bllip-lg-depth train.vec_dim=1024 train.n_heads=8 +train.dropout=0.1 +train.embedding_dropout=0.1 +train.output_dropout=0.1 train.lr=0.0001 train.use_stack_tape=True recursive_layer.attachment_decisions=True recursive_layer.rec_layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] recursive_layer.stack_pred_layer=16 +train.with_depth_info=True +recursive_layer.compose_keys_and_stack_info=True train.save_dir=path/to/save/dir
```

To train a model without pushdown layers, but with syntactic supervision, run:
```
python train_transformers.py train.encoder_n_layers=16 train.dataset=bllip-lg-depth train.vec_dim=1024 train.n_heads=8 +train.dropout=0.1 +train.embedding_dropout=0.1 +train.output_dropout=0.1 train.lr=0.0001 train.save_dir= train.save_dir=path/to/save/dir
```



## Inference / Evaluation



### Evaluating Dyck Generalization
To run the depth / length generalization eval from the paper, run the script: `eval_utils/eval_dyck.py`. Here is one example

```
cd eval_utils
python eval_dyck.py model_name=/path/to/save/dir/ckpt.pickle eval_type=depth 
```

### Preprocessed BLIMP (for evaluation)

### SyntaxGym (for surprisal evaluation)


### Evaluating on BLLIP

### Evaluating Surprisals on SyntaxGym