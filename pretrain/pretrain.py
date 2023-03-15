# Reference: https://huggingface.co/course/chapter7/3?fw=pt
import os, sys
sys.path.append('..')
import argparse
from tokenizer import BertTokenizer
from bert import BertModel
from datasets import load_dataset # hugging face
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='./data/sst-quora-sts-pretrain-train.csv')
    parser.add_argument('--test_data_path', default='./data/sst-quora-sts-pretrain-test.csv') 
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()


    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # load train, test dataset texts
    dataset = load_dataset("csv", data_files={'train': args.train_data_path, 'test': args.test_data_path}, delimiter='\t\t') # 'text', contains \theta, \times...

    # tokenize text
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"]) # 'input_ids', 'token_type_ids', 'attention_mask'
    
    # concatentate and chop data into length 128 chunks
    chunk_size = 128
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    # shuffle dataset
    lm_datasets = lm_datasets.shuffle(seed=42)
    
    # random masking for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    batch_size = 16 # 64
    # Show the training loss with every epoch
    logging_steps = len(lm_datasets["train"]) // batch_size

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        push_to_hub=False,
        fp16=True,
        logging_steps=logging_steps,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    trainer.train()
    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


