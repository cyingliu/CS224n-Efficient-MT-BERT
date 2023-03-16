# Get init embedding for prefix tuning
# The init embedding is the average (over batch, length) of hidden states at each layer
# Save the weights to weights/
import argparse
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from bert import BertModel
from config import BertConfig
from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data
from multitask_classifier import MultitaskBERT, get_args
import json
from tokenizer import BertTokenizer


if __name__ == "__main__":
    args = get_args()
    args.config_path = 'config/prefix_getinit_config.json'
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    loader = iter(sst_dev_dataloader)

    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': len(num_labels),
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'concat_pair': args.concat_pair,
              'config_path': args.config_path,
              'downstream': args.downstream,
              'similarity_classifier_type': args.similarity_classifier_type,
              'pooling_type': args.pooling_type,
              'sentiment_pooling_type': args.sentiment_pooling_type,
              'classification_concat_type': args.classification_concat_type,
              'pretrained_path': args.pretrained_path}
    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    batch = next(loader)
    b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
    b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
    logits = model.predict_sentiment(b_ids, b_mask)

