#! /bin/bash

# Finetune the model
python multitask_classifier.py --option pretrain --use_gpu\
	--output_dir result/prefix_pretrain_20_lr4e-3\
    --epochs 25 --lr 4e-3 --batch_size 16 --steps_per_epoch 2400 --eval_interval 4\
    --gradient_accumulation_step 1\
    --hidden_dropout_prob 0.3\
    --sample anneal\
    --config_path config/prefix_config.json
