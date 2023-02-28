import csv

import torch
from torch.utils.data import Dataset
from tokenizer import BertTokenizer

from datasets import preprocess_string

class SentencePairDataset_custom(Dataset):
    def __init__(self, dataset, args, isRegression =False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        sent = [x + self.tokenizer._sep_token + y for (x, y) in zip(sent1, sent2)]

        encoding = self.tokenizer(sent, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        token_type_ids = torch.LongTensor(encoding['token_type_ids'])

        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)
            

        return (token_ids, token_type_ids, attention_mask,
                labels,sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         labels, sent_ids) = self.pad_data(all_data)
        # keep the format same as default dataset
        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids,
                'token_type_ids_2': token_type_ids,
                'attention_mask_2': attention_mask,
                'labels': labels,
                'sent_ids': sent_ids
            }

        return batched_data

class SentencePairTestDataset_custom(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        sent = [x + self.tokenizer._sep_token + y for (x, y) in zip(sent1, sent2)]

        encoding = self.tokenizer(sent, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        token_type_ids = torch.LongTensor(encoding['token_type_ids'])
        
        return (token_ids, token_type_ids, attention_mask,
               sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids,
                'token_type_ids_2': token_type_ids,
                'attention_mask_2': attention_mask,
                'sent_ids': sent_ids
            }

        return batched_data

if __name__ == "__main__":
	from torch.utils.data import DataLoader
	from datasets import load_multitask_data
	from multitask_classifier import get_args

	args = get_args()
	sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

	dataset = SentencePairDataset_custom(para_dev_data, args, isRegression=True)
	dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=dataset.collate_fn)
	dataloader = iter(dataloader)
	batch = dataloader.next()

	token_ids = batch['token_ids']
	decoding = dataset.tokenizer.batch_decode(token_ids)
	print(decoding[0])
	

