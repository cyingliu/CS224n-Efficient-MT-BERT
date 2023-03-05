#! /bin/bash

# Finetune the model
python multitask_classifier.py --option finetune\
	--output_dir result/tmp \
    --epochs 25 --lr 1e-5 --batch_size 16 --steps_per_epoch 2400 --eval_interval 4\
    --gradient_accumulation_step 1\
    --hidden_dropout_prob 0.3\
    --sample squareroot\
    --config_path config/pal_config.json