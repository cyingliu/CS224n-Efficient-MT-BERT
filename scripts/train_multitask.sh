#! /bin/bash

# Finetune the model
python multitask_classifier.py --option finetune --use_gpu\
	--output_dir result/p2_finetune_baseline \
    --epochs 10 --lr 1e-5 --batch_size 16\
    --hidden_dropout_prob 0.3\
    --sample rr