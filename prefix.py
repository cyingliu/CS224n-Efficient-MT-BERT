import os
import torch
import torch.nn as nn

class Prefix(nn.Module):
    def __init__(self, config, layer_id, task_id):
        super(Prefix, self).__init__()
        if isinstance(config.prefix_length, list):
            self.prefix_length = config.prefix_length[task_id]
        else:
            self.prefix_length = config.prefix_length
        if config.prefix_init:
            emb = torch.load(os.path.join('weights', f'emb_{layer_id}.pt')) # (768)
            print(f"Load embedding weight from {os.path.join('weights', f'emb_{layer_id}.pt')}")
            emb = emb.view(1, 1, config.hidden_size).repeat(1, self.prefix_length, 1)
            self.prefix = nn.Parameter(emb)
        else:
            self.prefix = nn.Parameter(torch.ones((1, self.prefix_length, config.hidden_size)))
        
        self.layer_id = layer_id

    def forward(self, hidden_states, attention_mask=None):
        bs = hidden_states.shape[0]
        device = hidden_states.device

        if self.layer_id > 0:
            hidden_states = hidden_states[:, self.prefix_length:, :]

        prefix_expanded = self.prefix.expand(bs, -1, -1)
        hidden_states = torch.cat((prefix_expanded, hidden_states), dim=1)
        
        if attention_mask is not None:
            prefix_attention_mask = torch.zeros((bs, 1, 1, self.prefix_length), dtype=torch.bool).to(device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=-1)

        return hidden_states, attention_mask
            



