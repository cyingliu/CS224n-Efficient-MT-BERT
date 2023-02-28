#! /bin/bash

# Finetune the model
python multitask_classifier.py --option pretrain\
	--output_dir result/tmp \
    --epochs 6 --lr 1e-3 --batch_size 16 \
    --hidden_dropout_prob 0.3\
    --sample rr --concat_pair