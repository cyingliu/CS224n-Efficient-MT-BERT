import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask

from itertools import cycle
import yaml


TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        raise NotImplementedError


    def forward(self, input_ids, attention_mask):
        # 'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        raise NotImplementedError


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        raise NotImplementedError


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        raise NotImplementedError


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        raise NotImplementedError




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sst_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)
    
    # id2tasks = {0: "sst", 1: "para", 2: "sts"}
    train_loaders = [cycle(iter(sst_train_dataloader)), 
                     cycle(iter(para_train_dataloader)),
                     cycle(iter(sts_train_dataloader))]

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    sst_best_dev_acc = 0
    para_best_dev_acc = 0
    sts_best_dev_acc = 0
    avg_best_dev_acc = 0

    train_log_dir = os.path.join(args.output_dir, "log", "train")
    val_log_dir = os.path.join(args.output_dir, "log", "val")
    if not os.exists(train_log_dir):
        os.makedirs(train_log_dir)
    if not os.exists(val_log_dir):
        os.makedirs(val_log_dir)
    train_writer = SummaryWriter(log_dir=train_log_dir)
    val_writer = SummaryWriter(log_dir=val_log_dir)

    # Run for the specified number of epochs
    steps_per_epoch = len(sst_train_dataloader) + len(para_train_dataloader) + len(sts_train_dataloader)
    task_id = 0
    step = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = [0. for i in range(3)]
        num_batches = [0 for i in range(3)]
        for _ in range(steps_per_epoch):
            if args.sample == 'rr':
                task_id = (task_id + 1) % 3
            else:
                # TODO aneal sampling
                raise ValueError(f"Invalid sample method: {args.sample}")
            
            batch = next(train_loaders[task_id])

            if task_id == 0: # sst
                b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
                b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
            elif task_id == 1: # para
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels \
                    = (batch['token_ids_1'], batch['attention_mask_1'],\
                       batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels \
                    = b_ids.to(device), b_mask1.to(device),\
                      b_ids2.to(device), b_mask2.to(device), b_labels.to(device)
                logits = model.predict_paraphrase(b_ids, b_mask1, b_ids2, b_mask2)
                loss = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1), reduction='sum') / args.batch_size
            elif task_id == 2: # sts
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels \
                    = (batch['token_ids_1'], batch['attention_mask_1'],\
                       batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels \
                    = b_ids.to(device), b_mask1.to(device),\
                      b_ids2.to(device), b_mask2.to(device), b_labels.to(device)
                logits = model.predict_similarity(b_ids, b_mask1, b_ids2, b_mask2)
                loss = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1), reduction='sum') / args.batch_size
            else:
                raise ValueError(f"Invalid task_id: {task_id}")
          
            # TODO gradient accumulation
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            train_loss[task_id] += loss.item()
            num_batches[task_id] += 1

        for task_id in range(3):
            train_loss[task_id] = train_loss[task_id] / num_batches[task_id]

        para_train_acc, _, _, sst_train_acc, _, _, sts_train_acc, _, _ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        para_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_acc, _, _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        avg_dev_acc = np.mean([para_dev_acc, sst_dev_acc, sts_dev_acc])

        
        if sst_dev_acc > sst_best_dev_acc:
            sst_best_dev_acc = sst_dev_acc
            save_model(model, optimizer, args, config, os.path.join(args.output_dir, 'best-sst-multi-task-classifier.pt'))
        if para_dev_acc > para_best_dev_acc:
            para_best_dev_acc = para_dev_acc
            save_model(model, optimizer, args, config, os.path.join(args.output_dir, 'best-para-multi-task-classifier.pt'))
        if sts_dev_acc > sst_best_dev_acc:
            sts_best_dev_acc = sts_dev_acc
            save_model(model, optimizer, args, config, os.path.join(args.output_dir, 'best-sts-multi-task-classifier.pt'))
        if avg_dev_acc > avg_best_dev_acc:
            avg_best_dev_acc = avg_dev_acc
            save_model(model, optimizer, args, config, os.path.join(args.output_dir, 'best-avg-multi-task-classifier.pt'))
        
        train_writer.add_scalar("sst_loss", train_loss[0], epoch)
        train_writer.add_scalar("para_loss", train_loss[1], epoch)
        train_writer.add_scalar("sts_loss", train_loss[2], epoch)
        train_writer.add_scalar("sst_acc", sst_train_acc, epoch)
        train_writer.add_scalar("para_acc", para_train_acc, epoch)
        train_writer.add_scalar("sts_acc", sts_train_acc, epoch)
        val_writer.add_scalar("sst_acc", sst_dev_acc, epoch)
        val_writer.add_scalar("para_acc", para_dev_acc, epoch)
        val_writer.add_scalar("sts_acc", sst_dev_acc, epoch)

        print(f"Epoch {epoch}: train sst loss :: {train_loss[0] :.3f}, train para loss :: {train_loss[1] :.3f}, train sts loss :: {train_loss[2] : .3f},\n\
                train sst acc :: {sst_train_acc :.3f}, train para acc :: {para_train_acc}, train sts acc :: {sts_train_acc},\n\
                dev sst acc :: {sst_dev_acc :.3f}, dev para acc :: {para_dev_acc}, dev sts acc :: {sts_dev_acc}")



def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(os.path.join(args.output_dir, 'best-avg-multi-task-classifier.pt'))
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {os.path.join(args.output_dir, 'best-avg-multi-task-classifier.pt')}")

        test_model_multitask(args, model, device)


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

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    # output
    parser.add_argument("--output_dir", "--output_dir", type=str, help="dir for saved model (.pt) and prediction files (.csv)",
                        default="result/tmp")
    # multi-task
    parser.add_argument("--sample", help='sample method for multi dataset (rr)', type=str, default='rr')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "config.yaml"), 'w') as outfile:
        yaml.dump(vars(args), outfile)
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
