#! /bin/bash

# Finetune the model
python classifier.py --option finetune --use_gpu \
	--output_dir result/p1_finetune \
        --epochs 10 --lr 1e-5 --batch_size 8\
        --hidden_dropout_prob 0.3

# Pretrain the model
python classifier.py --option pretrain --use_gpu \
		--output_dir result/p1_pretrain \
		--epochs 10 --lr 1e-3 --batch_size 8\
		--hidden_dropout_prob 0.3
