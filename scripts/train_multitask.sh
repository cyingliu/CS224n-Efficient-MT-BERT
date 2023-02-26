#! /bin/bash

# Finetune the model
python multitask_classifier.py --option finetune \
	--output_dir result/tmp \
    --epochs 10 --lr 1e-5 --batch_size 8\
    --hidden_dropout_prob 0.3\
    --sample rr