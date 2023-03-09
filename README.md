# CS 224N Default Final Project - Multitask BERT

This is the starting code for the default final project for the Stanford CS 224N class. You can find the handout [here](https://web.stanford.edu/class/cs224n/project/default-final-project-bert-handout.pdf)

In this project, you will implement some important components of the BERT model to better understanding its architecture. 
You will then use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection and semantic similarity.

After finishing the BERT implementation, you will have a simple model that simultaneously performs the three tasks.
You will then implement extensions to improve on top of this baseline.

## Setup instructions

* Follow `setup.sh` to properly setup a conda environment and install dependencies.
* There is a detailed description of the code structure in [STRUCTURE.md](./STRUCTURE.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, external libraries that give you other pre-trained models or embeddings are not allowed (e.g., `transformers`).

## Handout

Please refer to the handout for a through description of the project and its parts.

### Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
## Usage
```
python multitask_classifier.py --option [finetune/pretrain] --use_gpu\
	  --output_dir OUTPUT_DIR\
    --epochs 25 --lr 1e-5 --lr_adapt 1e-4 --warmup_portion 0.1\
    --batch_size 16 --steps_per_epoch 2400 --eval_interval 4\
    --gradient_accumulation_step 1\
    --hidden_dropout_prob 0.3\
    --sample [rr, squareroot, anneal]\
    --config_path CONFIG_PATH
```
## Other
Paper link: https://docs.google.com/spreadsheets/d/1LWrbaXWh6i8SJbvJ5o-TWr5RkVuHSKxputMlH6aMZVQ/edit?usp=sharing. 

Experiment table link: https://docs.google.com/spreadsheets/d/1Gsw49kSUKG4NlRpbCfMYUcfCjo6qEvHWP7uNDIKc1V4/edit?usp=sharing

Hackmd link: https://hackmd.io/@grcgs2212/By2_e6R2o

PAL (multi-task, houlsby) reference code: https://github.com/AsaCooperStickland/Bert-n-Pals
