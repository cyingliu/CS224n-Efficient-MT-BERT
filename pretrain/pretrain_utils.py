import torch
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_pretrained_weight_from_lm(model, checkpoint_path):

    print(f"Loading pretrained weight from {checkpoint_path}")
    loads = torch.load(checkpoint_path, map_location=device)
    
    lm2bert = json.load(open('pretrain/lm2bert.json'))
    # lm2bert = {}
    # for name1, name2 in zip(list(model.state_dict().keys()), list(loads.keys())):
    #     if 'cls' in name2: continue # LM head
    #     lm2bert[name2] = name1
    
    # with open("lm2bert.json", "w") as outfile:
    #     json.dump(lm2bert, outfile, indent=2)

    updated_loads = {}
    for name, param in loads.items():
        if 'cls' in name: continue # LM head
        updated_loads[lm2bert[name]] = param

    model.load_state_dict(updated_loads, strict=False) # miss 'pooler_dense.weight', 'pooler_dense.bias', LM BERT don't have these weights, train them from scratch

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from bert import BertModel
    from config import BertConfig
    
    bert_config = BertConfig.from_pretrained('bert-base-uncased')
    bert_config.name_or_path = "from_scratch"
    model = BertModel(bert_config) #
    load_pretrained_weight_from_lm(model, "pretrain-sst-quora-sts/pytorch_model.bin")


