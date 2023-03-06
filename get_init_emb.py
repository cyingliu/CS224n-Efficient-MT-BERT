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
from multitask_classifier import MultitaskBERT
import json
from tokenizer import BertTokenizer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")
    
    parser.add_argument("--use_gpu", action='store_true')

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--config_path", help='config (.json) file for adaptation modules', type=str, default="config/prefix_getinit_config.json")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
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
              'option': 'finetune',
              'concat_pair': False,
              'config_path': args.config_path}
    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    batch = next(loader)
    b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
    b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
    logits = model.predict_sentiment(b_ids, b_mask)

