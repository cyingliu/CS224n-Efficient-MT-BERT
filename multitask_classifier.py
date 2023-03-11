import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bert import BertModel
from config import BertConfig
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data
from datasets_custom import SentencePairDataset_custom, SentencePairTestDataset_custom

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask

from itertools import cycle
import yaml
import json
from tokenizer import BertTokenizer


TQDM_DISABLE=False

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

def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Reference: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model/62764464#62764464
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


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
        if config.config_path: # extra config for adaptation modules
            with open(config.config_path) as f:
                _config = json.load(f)
            bert_config = BertConfig.from_pretrained('bert-base-uncased', **_config)
            self.bert = BertModel.from_pretrained('bert-base-uncased', config=bert_config)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        

        ##### Adaptation Modules #####
        # Load pretrained layernorm weight/bias for houlsby task-specific layernorm

        if config.config_path and bert_config.houlsby and bert_config.houlsby_add_layernorm:
            print("Intialize houlsby layer norm")
            state_dict = self.bert.state_dict()
            update_state_dict = {}
            for layer in range(bert_config.num_hidden_layers):
                weight_0 = state_dict[f'bert_layers.{layer}.attention_layer_norm.weight']
                bias_0 = state_dict[f'bert_layers.{layer}.attention_layer_norm.bias']
                weight_1 = state_dict[f'bert_layers.{layer}.out_layer_norm.weight']
                bias_1 = state_dict[f'bert_layers.{layer}.out_layer_norm.bias']
                for task in range(bert_config.num_tasks):
                    update_state_dict[f'bert_layers.{layer}.houlsbys_0.{task}.layernorm.weight'] = weight_0.clone()
                    update_state_dict[f'bert_layers.{layer}.houlsbys_0.{task}.layernorm.bias'] = bias_0.clone()
                    update_state_dict[f'bert_layers.{layer}.houlsbys_1.{task}.layernorm.weight'] = weight_1.clone()
                    update_state_dict[f'bert_layers.{layer}.houlsbys_1.{task}.layernorm.bias'] = bias_1.clone()
        state_dict.update(update_state_dict)
        self.bert.load_state_dict(state_dict)
        self.bert.state_dict()[f'bert_layers.0.houlsbys_0.0.layernorm.weight'] = self.bert.state_dict()[f'bert_layers.0.houlsbys_0.0.layernorm.weight'] + 1
        
        ##############################
        for name, param in self.bert.named_parameters():
            if config.option == 'pretrain':
                if 'pal' in name or 'prefix' in name or 'houlsby' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        self.concat_pair = config.concat_pair
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        n = 1 if self.concat_pair else 2
        if config.downstream == 'single':
            self.sentiment_classifier = nn.Linear(config.hidden_size, config.num_labels) 
            self.paraphrase_classifier = nn.Linear(config.hidden_size * n, 1)
            self.similarity_classifier = nn.Linear(config.hidden_size * n, 1)
        elif config.downstream == 'double':
            self.sentiment_classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.num_labels)
            )
            self.paraphrase_classifier = nn.Sequential(
                nn.Linear(config.hidden_size * n, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 1)
            )
            self.similarity_classifier = nn.Sequential(
                nn.Linear(config.hidden_size * n, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 1)
            )
        else:
            raise ValueError(f"Invalid downstream: {config.downstream}")


    def forward(self, input_ids, attention_mask, task_id=0):
        # 'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        outputs = self.bert(input_ids, attention_mask, task_id=task_id)
        sequence_output, pooled_output = outputs['last_hidden_state'], outputs['pooler_output']
        return sequence_output, pooled_output


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        sequence_output, pooled_output = self.forward(input_ids, attention_mask, task_id=0)
        pooled_output = self.dropout(pooled_output)
        logits = self.sentiment_classifier(pooled_output)

        return logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        if self.concat_pair:
            sequence_output, pooled_output = self.forward(input_ids_1, attention_mask_1, task_id=1)
            pooled_output = self.dropout(pooled_output)
        
        else:
            sequence_output_1, pooled_output_1 = self.forward(input_ids_1, attention_mask_1, task_id=1)
            sequence_output_2, pooled_output_2 = self.forward(input_ids_2, attention_mask_2, task_id=1)
            pooled_output_1 = self.dropout(pooled_output_1)
            pooled_output_2 = self.dropout(pooled_output_2)
            pooled_output = torch.cat((pooled_output_1, pooled_output_2), dim=-1)
        
        logits = self.paraphrase_classifier(pooled_output)
        
        return logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        if self.concat_pair:
            sequence_output, pooled_output = self.forward(input_ids_1, attention_mask_1, task_id=2)
            pooled_output = self.dropout(pooled_output)
        
        else:
            sequence_output_1, pooled_output_1 = self.forward(input_ids_1, attention_mask_1, task_id=2)
            sequence_output_2, pooled_output_2 = self.forward(input_ids_2, attention_mask_2, task_id=2)
            pooled_output_1 = self.dropout(pooled_output_1)
            pooled_output_2 = self.dropout(pooled_output_2)
            pooled_output = torch.cat((pooled_output_1, pooled_output_2), dim=-1)
        
        logits = self.similarity_classifier(pooled_output)

        return logits


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
    print("Devcie:", device)
    if args.concat_pair:
        sentencepair_dataset = SentencePairDataset_custom
    else:
        sentencepair_dataset = SentencePairDataset

    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_data = sentencepair_dataset(para_train_data, args)
    para_dev_data = sentencepair_dataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_dev_data.collate_fn)

    sts_train_data = sentencepair_dataset(sts_train_data, args, isRegression=True)
    sts_dev_data = sentencepair_dataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)
    
    # id2tasks = {0: "sst", 1: "para", 2: "sts"}
    train_loaders = [cycle(iter(sst_train_dataloader)), 
                     cycle(iter(para_train_dataloader)),
                     cycle(iter(sts_train_dataloader))]
    dataset_lengths = [len(sst_train_data), len(para_train_data), len(sts_train_data)] # [8544, 141498, 6040]
    loss_weight = {0: args.weight_sst, 1: args.weight_para, 2: args.weight_sts}

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': len(num_labels),
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'concat_pair': args.concat_pair,
              'config_path': args.config_path,
              'downstream': args.downstream}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    # Print total trainable parameters
    total_params = numel(model, only_trainable=True)
    print("Total trainable params:", total_params)

    ##### Optimizer/Scheduler #####
    lr = args.lr
    # adaptation_modules = ['pal', 'prefix', 'houlsby', 'classifier']
    backbone_params = []
    adaptation_params = []
    for name, param in model.named_parameters():
        if 'pal' in name or 'prefix' in name or 'houlsby' in name or 'classifier' in name:
            adaptation_params.append(param)
        else:
            backbone_params.append(param)
    optimizer = AdamW([{'params': adaptation_params, 'lr': args.lr_adapt}, {'params': backbone_params}], lr=lr)
    
    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_portion)
    def warmup(current_step: int):
        if current_step < warmup_steps:  # current_step / warmup_steps * base_lr
            return float(current_step / warmup_steps)
        else:                                 # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
    #####################
    sst_best_dev_acc = 0
    para_best_dev_acc = 0
    sts_best_dev_acc = 0
    avg_best_dev_acc = 0

    train_log_dir = os.path.join(args.output_dir, "log", "train")
    val_log_dir = os.path.join(args.output_dir, "log", "val")
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    if not os.path.exists(val_log_dir):
        os.makedirs(val_log_dir)
    train_writer = SummaryWriter(log_dir=train_log_dir)
    val_writer = SummaryWriter(log_dir=val_log_dir)

    # Run for the specified number of epochs
    # steps_per_epoch = len(sst_train_dataloader) + len(para_train_dataloader) + len(sts_train_dataloader) # 9756
    steps_per_epoch = args.steps_per_epoch

    task_id = 0
    step = 1
    train_loss = [0. for i in range(3)]
    num_batches = [0 for i in range(3)]
    tr_sst_loss, tr_para_loss, tr_sts_loss = None, None, None
    
    tot = np.sum(dataset_lengths)
    Nprobs = [p/tot for p in dataset_lengths]
    Sqprobs = np.sqrt(dataset_lengths)
    tot = np.sum(Sqprobs)
    Sqprobs = [p/tot for p in Sqprobs]
    
    for epoch in range(args.epochs):
        model.train()
        
        for _ in tqdm(range(steps_per_epoch), desc=f'train-{epoch}', disable=TQDM_DISABLE):
            if args.sample == 'rr':
                task_id = (task_id + 1) % 3
            elif args.sample == "proportional":
                probs = Nprobs
                task_id = np.random.choice(3, p=probs)
            elif args.sample == "squareroot":
                probs = Sqprobs
                task_id = np.random.choice(3, p=probs)
            elif args.sample == "anneal":
                alpha = 1 - 0.8 * epoch / (args.epochs - 1)
                aprobs = np.power(dataset_lengths, alpha)
                tot = np.sum(aprobs)
                probs = aprobs / tot
                task_id = np.random.choice(3, p=probs)
            else:
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
                    = b_ids1.to(device), b_mask1.to(device),\
                      b_ids2.to(device), b_mask2.to(device), b_labels.to(device)

                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), b_labels.float(), reduction='sum') / args.batch_size
            elif task_id == 2: # sts
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels \
                    = (batch['token_ids_1'], batch['attention_mask_1'],\
                       batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels \
                    = b_ids1.to(device), b_mask1.to(device),\
                      b_ids2.to(device), b_mask2.to(device), b_labels.to(device)
                logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                loss = F.mse_loss(logits.squeeze(-1), b_labels.float(), reduction='sum') / args.batch_size
            else:
                raise ValueError(f"Invalid task_id: {task_id}")
          
            # gradient accumulation
            loss = loss_weight[task_id] * loss / args.gradient_accumulation_step
            loss.backward()

            train_loss[task_id] += loss.item()
            num_batches[task_id] += 1

            
            if (step + 1) % args.gradient_accumulation_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % args.log_interval == 0:
                for task_id in range(3):
                    if num_batches[task_id] == 0:
                        train_loss[task_id] = float('nan')
                    else:
                        train_loss[task_id] = train_loss[task_id] / num_batches[task_id]
                tot = np.sum(num_batches)
                sample_prob = [n/tot for n in num_batches]

                train_writer.add_scalar("sst_loss", train_loss[0], step)
                train_writer.add_scalar("para_loss", train_loss[1], step)
                train_writer.add_scalar("sts_loss", train_loss[2], step)
                train_writer.add_scalar("sst_sample", sample_prob[0], step)
                train_writer.add_scalar("para_sample", sample_prob[1], step)
                train_writer.add_scalar("sts_sample", sample_prob[2], step)
                train_writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
                
                tr_sst_loss, tr_para_loss, tr_sts_loss = train_loss[0], train_loss[1], train_loss[2]
                train_loss = [0. for i in range(3)]
                num_batches = [0 for i in range(3)]
            step += 1

        if (epoch + 1) % args.eval_interval == 0:
            para_train_acc, _, _, sst_train_acc, _, _, sts_train_acc, _, _ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
            avg_train_acc = np.mean([para_train_acc, sst_train_acc, sts_train_acc])
        para_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_acc, _, _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        avg_dev_acc = np.mean([para_dev_acc, sst_dev_acc, sts_dev_acc])
   
        if sst_dev_acc > sst_best_dev_acc:
            sst_best_dev_acc = sst_dev_acc
            # save_model(model, optimizer, args, config, os.path.join(args.output_dir, 'best-sst-multi-task-classifier.pt'))
        if para_dev_acc > para_best_dev_acc:
            para_best_dev_acc = para_dev_acc
            # save_model(model, optimizer, args, config, os.path.join(args.output_dir, 'best-para-multi-task-classifier.pt'))
        if sts_dev_acc > sts_best_dev_acc:
            sts_best_dev_acc = sts_dev_acc
            # save_model(model, optimizer, args, config, os.path.join(args.output_dir, 'best-sts-multi-task-classifier.pt'))
        if avg_dev_acc > avg_best_dev_acc:
            avg_best_dev_acc = avg_dev_acc
            save_model(model, optimizer, args, config, os.path.join(args.output_dir, 'best-avg-multi-task-classifier.pt'))
        
        
        # log train status
        if (epoch + 1) % args.eval_interval == 0:
            train_writer.add_scalar("sst_acc", sst_train_acc, step)
            train_writer.add_scalar("para_acc", para_train_acc, step)
            train_writer.add_scalar("sts_acc", sts_train_acc, step)
            train_writer.add_scalar("avg_acc", avg_train_acc, step)

        val_writer.add_scalar("sst_acc", sst_dev_acc, step)
        val_writer.add_scalar("para_acc", para_dev_acc, step)
        val_writer.add_scalar("sts_acc", sts_dev_acc, step)
        val_writer.add_scalar("avg_acc", avg_dev_acc, step)

        if (epoch + 1) % args.eval_interval == 0:
            print(f"Epoch {epoch}: train sst loss :: {tr_sst_loss :.3f}, train para loss :: {tr_para_loss :.3f}, train sts loss :: {tr_sts_loss : .3f},\n\
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
    parser.add_argument("--steps_per_epoch", type=int, default=2400, help="PAL paper value: 2400, sum of len(data loader): 9756")
    parser.add_argument("--eval_interval", type=int, default=4, help="Epoch interval for evaluation through whole train set")
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
    parser.add_argument("--lr_adapt", type=float, help="lr for pal, houlsby, prefix, and classifier", default=1e-3)
    parser.add_argument("--warmup_portion", type=float, default=0.1, help="linear warmup portion")
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    # training setting
    parser.add_argument("--output_dir", type=str, help="dir for saved model (.pt) and prediction files (.csv)",
                        default="result/tmp")
    parser.add_argument("--log_interval", type=int, help="interval for log writer", default=100)
    # multi-task
    parser.add_argument("--sample", help='sample method for multi dataset', type=str, choices=('rr', 'proportional', 'squareroot', 'anneal'), default='rr')
    parser.add_argument("--weight_sst", help='weight for sst loss', type=float, default=1.0)
    parser.add_argument("--weight_para", help='weight for para loss', type=float, default=1.0)
    parser.add_argument("--weight_sts", help='weight for sts loss', type=float, default=1.0)
    # model architecture
    parser.add_argument("--downstream", help="single/double linear downstream classifiers", type=str, choices=('single', 'double'), default='single')
    parser.add_argument("--config_path", help='config (.json) file for adaptation modules', type=str, default="")
    # dataset
    parser.add_argument("--concat_pair", action='store_true', help="concat two sequences if True, feed separately if False")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    ### Save experiment config ###
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "config.yaml"), 'w') as outfile:
        yaml.dump(vars(args), outfile)
    if args.config_path:
        with open(args.config_path) as f:
                config = json.load(f)
        with open(os.path.join(args.output_dir, "model_config.json"), 'w') as f:
            json.dump(config, f)
    ### Define output paths ###
    args.sst_dev_out = os.path.join(args.output_dir, "sst-dev-output.csv")
    args.sst_test_out = os.path.join(args.output_dir, "sst-test-output.csv")
    args.para_dev_out = os.path.join(args.output_dir, "para-dev-output.csv")
    args.para_test_out = os.path.join(args.output_dir, "para-test-output.csv")
    args.sts_dev_out = os.path.join(args.output_dir, "sts-dev-output.csv")
    args.sts_test_out = os.path.join(args.output_dir, "sts-test-output.csv")
    ############################
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
