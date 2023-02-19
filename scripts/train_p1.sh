#! /bin/bash

# Finetune the model
python classifier.py --option finetune \
        --epochs 10 --lr 1e-5 --batch_size 8\
        --hidden_dropout_prob 0.3