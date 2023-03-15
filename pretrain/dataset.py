import os, sys
sys.path.append('..')

import csv
import torch
from datasets import Dataset # huggingface

from tokenizer import BertTokenizer



def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())

class TextDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        data = self.load_data()
        self.text = data
        # self.tokenized_data = self.tokenize_data()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]

    def load_data(self):
        print("Loading data ...")
        fin = open(self.filename, 'r')
        lines = fin.readlines()
        data = [preprocess_string(s) for s in lines]
        return data
    # def tokenize_data(self):
    #     print("Tokenizing data ...")
    #     tokenized_data = []
    #     for sent in self.text:
    #         encoding = self.tokenizer(sent, return_tensors='pt')
    #         token_ids = torch.LongTensor(encoding['input_ids'])
    #         tokenized_data.append(token_ids.squeeze(0))
    #     return tokenized_data


