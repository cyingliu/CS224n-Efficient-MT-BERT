import torch
import torch.nn as nn
import torch.nn.functional as F

class Houlsby(nn.Module):
    def __init__(self, config):
        super(Houlsby, self).__init__()
        
        self.down_project = nn.Linear(config.hidden_size, config.houlsby_size)
        self.up_project = nn.Linear(config.houlsby_size, config.hidden_size)
        self.act_fn = F.relu

        if config.houlsby_add_layernorm:
        	self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        output = self.down_project(hidden_states)
        output = self.act_fn(output)
        output = self.up_project(output)

        return output