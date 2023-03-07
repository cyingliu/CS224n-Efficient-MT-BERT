#! /bin/bash

# Finetune the model
python test_multitask.py --use_gpu \
--output_dir result/pal_pretrain_share_204_lr1e-3 \
--config_path result/pal_pretrain_share_204_lr1e-3/model_config.json