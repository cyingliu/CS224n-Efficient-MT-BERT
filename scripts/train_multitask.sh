#! /bin/bash

# Finetune the model
python multitask_classifier.py --option pretrain --use_gpu\
	--output_dir result/pal_pretrain_share_204_lr1e-4\
    --epochs 25 --lr 1e-4 --batch_size 16 --steps_per_epoch 2400 --eval_interval 4\
    --gradient_accumulation_step 1\
    --hidden_dropout_prob 0.3\
    --sample anneal\
    --config_path config/pal_config.json
