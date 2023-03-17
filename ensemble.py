import os
import yaml
import pandas as pd
import argparse
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
from tqdm import tqdm
import numpy as np

from multitask_classifier import MultitaskBERT
from datasets import load_multitask_data, load_multitask_test_data, \
    SentenceClassificationDataset, SentenceClassificationTestDataset, \
    SentencePairDataset, SentencePairTestDataset

TQDM_DISABLE = False

def check_weights(df):
    columns = ['sentiment', 'paraphrase', 'semantic']

    for col in columns:
        assert sum(df[col]) == 1

def reset_args(model_args, args):
    model_args.sst_dev = args.sst_dev
    model_args.sst_test = args.sst_test
    model_args.para_dev = args.para_dev
    model_args.para_test = args.para_test
    model_args.sts_dev = args.sts_dev 
    model_args.sts_test = args.sts_test

def fix_config(config):
    default_config = {'hidden_dropout_prob': 0.1,
              'num_labels': 3,
              'hidden_size': 768,
              'data_dir': '.',
              'option': 'finetune',
              'concat_pair': False,
              'config_path': "",
              'downstream': 'double',
              'similarity_classifier_type': 'cosine-similarity',
              'pooling_type': 'mean',
              'sentiment_pooling_type': 'mean',
              'classification_concat_type':'add-abs',
              'pretrained_path': ""}

    for k in default_config:
        if k not in vars(config):
            setattr(config, k, default_config[k])
    
    return config

def test_model_multitask(args, models, device, df):
        
        sentencepair_dataset = SentencePairDataset
        sentencepair_testdataset = SentencePairTestDataset
        
        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = sentencepair_testdataset(para_test_data, args)
        para_dev_data = sentencepair_dataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = sentencepair_testdataset(sts_test_data, args) #
        sts_dev_data = sentencepair_dataset(sts_dev_data, args, isRegression=True) #

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        ##### Modification #####
        dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, dev_sts_corr, \
            dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, models, device, df)

        test_para_y_pred, test_para_sent_ids, test_sst_y_pred, \
            test_sst_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, models, device, df)

        ##### Modification #####
        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


# Perform model evaluation in terms by averaging accuracies across tasks.
def model_eval_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         models, device, df):
    for model in models:  
        model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():
        para_y_true = []
        para_y_pred = []
        para_sent_ids = []

        # Evaluate paraphrase detection.
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            y_hat = np.zeros(b_labels.flatten().shape)
            
            for i, model in enumerate(models):
                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat += df.iloc[i]['paraphrase'] * (logits.sigmoid().flatten().cpu().numpy())
            
            y_hat = y_hat.round()
            b_labels = b_labels.flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_y_true.extend(b_labels)
            para_sent_ids.extend(b_sent_ids)

        paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))

        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []


        # Evaluate semantic textual similarity.
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):

            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            y_hat = np.zeros(b_labels.shape)
            for i, model in enumerate(models):
                logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat += df.iloc[i]['semantic'] * (logits.flatten().cpu().numpy())
            
            b_labels = b_labels.flatten().cpu().numpy()
            sts_y_pred.extend(y_hat)
            sts_y_true.extend(b_labels)
            sts_sent_ids.extend(b_sent_ids)
        pearson_mat = np.corrcoef(sts_y_pred,sts_y_true)
        sts_corr = pearson_mat[1][0]


        sst_y_true = []
        sst_y_pred = []
        sst_sent_ids = []

        # Evaluate sentiment classification.
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_labels, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['labels'], batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            ensembled_logits = np.zeros((b_ids.shape[0], 5))
            for i, model in enumerate(models):
                logits = model.predict_sentiment(b_ids, b_mask)
                ensembled_logits += df.iloc[i]['sentiment'] * (logits.softmax(dim=1).cpu().numpy())
            
            y_hat = ensembled_logits.argmax(axis=-1).flatten()
            b_labels = b_labels.flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_y_true.extend(b_labels)
            sst_sent_ids.extend(b_sent_ids)

        sentiment_accuracy = np.mean(np.array(sst_y_pred) == np.array(sst_y_true))

        print(f'Paraphrase detection accuracy: {paraphrase_accuracy:.3f}')
        print(f'Sentiment classification accuracy: {sentiment_accuracy:.3f}')
        print(f'Semantic Textual Similarity correlation: {sts_corr:.3f}')

        return (paraphrase_accuracy, para_y_pred, para_sent_ids,
                sentiment_accuracy,sst_y_pred, sst_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids)

# Perform model evaluation in terms by averaging accuracies across tasks.
def model_eval_test_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         models, device, df):
    for model in models:
        model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():

        para_y_pred = []
        para_sent_ids = []
        # Evaluate paraphrase detection.
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            

            y_hat = np.zeros(b_ids1.shape[0])
            
            for i, model in enumerate(models):
                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat += df.iloc[i]['paraphrase'] * (logits.sigmoid().flatten().cpu().numpy())
            
            y_hat = y_hat.round()

            para_y_pred.extend(y_hat)
            para_sent_ids.extend(b_sent_ids)


        sts_y_pred = []
        sts_sent_ids = []


        # Evaluate semantic textual similarity.
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)


            y_hat = np.zeros(b_ids1.shape[0])
            for i, model in enumerate(models):
                logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat += df.iloc[i]['semantic'] * (logits.flatten().cpu().numpy())

            sts_y_pred.extend(y_hat)
            sts_sent_ids.extend(b_sent_ids)


        sst_y_pred = []
        sst_sent_ids = []

        # Evaluate sentiment classification.
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'],  batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            
            ensembled_logits = np.zeros((b_ids.shape[0], 5))
            for i, model in enumerate(models):
                logits = model.predict_sentiment(b_ids, b_mask)
                ensembled_logits += df.iloc[i]['sentiment'] * (logits.softmax(dim=1).cpu().numpy())
                
            y_hat = ensembled_logits.argmax(axis=-1).flatten()

            sst_y_pred.extend(y_hat)
            sst_sent_ids.extend(b_sent_ids)

        return (para_y_pred, para_sent_ids,
                sst_y_pred, sst_sent_ids,
                sts_y_pred, sts_sent_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_list', '-l', help='.csv file that contains model names, and optionally model weights', required=True)
    parser.add_argument('--result_dir', '-d', help='root directory for results', default='./result/')
    parser.add_argument('--output_dir', '-o', help='output directory for predicion files', default='./')
    
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    args = parser.parse_args()

    args.sst_dev_out = os.path.join(args.output_dir, "sst-dev-output.csv")
    args.sst_test_out = os.path.join(args.output_dir, "sst-test-output.csv")
    args.para_dev_out = os.path.join(args.output_dir, "para-dev-output.csv")
    args.para_test_out = os.path.join(args.output_dir, "para-test-output.csv")
    args.sts_dev_out = os.path.join(args.output_dir, "sts-dev-output.csv")
    args.sts_test_out = os.path.join(args.output_dir, "sts-test-output.csv")

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    df = pd.read_csv(args.model_list, delimiter='\t')
    models = []
    if len(df.columns) > 1:
        print("Use determined weights")
        check_weights(df)
        print(df.head())
        for i in range(len(df)):
            name = df.iloc[i]['name']
            model_dir = os.path.join(args.result_dir, name)
            
            saved = torch.load(model_dir, map_location=device)
            config = saved['model_config']
            if config.config_path:
                config.config_path = os.path.join(model_dir, 'model_config.json')
            config = fix_config(config)

            model = MultitaskBERT(config)
            model.load_state_dict(saved['model'])
            model = model.to(device)
            print(f"Loaded model to test from {model_dir}")
            models.append(model)

        test_model_multitask(args, models, device, df)



    else:
        print("Calculate weights")
        raise NotImplementedError
