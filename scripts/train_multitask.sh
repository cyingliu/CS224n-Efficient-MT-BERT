#! /bin/bash

# Finetune the model
python multitask_classifier.py --option finetune \
	--output_dir result/palDoubleLayer\
    --epochs 25 --lr 1e-5 --batch_size 16 --steps_per_epoch 2400\
    --gradient_accumulation_step 1\
    --hidden_dropout_prob 0.1\
    --sample anneal\
    --config_path config/pal_config.json