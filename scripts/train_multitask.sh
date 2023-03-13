#! /bin/bash

# Finetune the model
python multitask_classifier.py --option finetune --use_gpu\
	--output_dir result/multiheadBaseDataTests\
    --epochs 25 --lr 1e-5 --batch_size 16 --steps_per_epoch 2400 --eval_interval 5\
    --gradient_accumulation_step 1\
    --hidden_dropout_prob 0.1\
    --sample anneal\
