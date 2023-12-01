# Pushdown Transformer LMs
Code Repository for our EMNLP '23 paper: "Pushdown Layers: Encoding Recursive Structure in Transformer Language Models".

## Setup Environment
```
conda env create -f environment.yml
conda activate pushdown-lm
```

## Get Datasets

### Training on BLLIP

To get the `BLLIP-LG` datasets we use for training our models, please follow the instructions [here](https://github.com/IBM/transformers-struct-guidance). Once you have `BLLIP-LG` store it inside `data_utils` and run:

```
cd data_utils
python process_bllip.py 
```

This script tokenizes the dataset using the GPT2 tokenizer, and converts every parse tree into unlabeled binarized trees.

### Training on WikiTrees

WIP. Please send Shikhar Murty an email (can be found on the paper link) if you need the WikiTrees dataset.

## Quickstart (I just want to run inference)

Suppose you have some text, parsed according to the above pre-processing scheme.

## Training
Note: This repository uses WandB for logging. Make sure you set your WandB account before training.

### Pushdown LMs on Dyck

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

### Preprocessed BLIMP (for evaluation)

### SyntaxGym (for surprisal evaluation)

### Evaluating Dyck Generalization

### Evaluating on BLLIP

### Evaluating Surprisals on SyntaxGym