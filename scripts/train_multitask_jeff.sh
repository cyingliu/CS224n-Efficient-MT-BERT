#! /bin/bash

# Finetune the model
python multitask_classifier.py --option finetune --use_gpu\
	--output_dir result/palDoubleLayer6Heads408Size\
  --epochs 25 --lr 1e-5 --batch_size 16 --steps_per_epoch 2400 --eval_interval 8\
  --gradient_accumulation_step 1\
  --hidden_dropout_prob 0.1\
  --sample anneal\
  --config_path config/pal_config.json\
  --lr_adapt 1e-5\
  --warmup_portion .1
