#! /bin/bash

# Finetune the model
python multitask_classifier.py --option finetune\
	--output_dir result/tmp \
    --epochs 7 --lr 1e-3 --batch_size 16 --steps_per_epoch 2400\
    --gradient_accumulation_step 2\
    --hidden_dropout_prob 0.3\
    --sample anneal