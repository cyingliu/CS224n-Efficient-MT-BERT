import torch
import torch.nn as nn

class PAL(nn.Module):
    

class BERTPals(nn.Module):
    def __init__(self, config, extra_dim=None):
        super(BERTPals, self).__init__()
        # Encoder and decoder matrices project down to the smaller dimension
        self.aug_dense = nn.Linear(config.hidden_size, config.hidden_size_aug)
        self.aug_dense2 = nn.Linear(config.hidden_size_aug, config.hidden_size)
        # Attention without the final matrix multiply.
        self.attn = BERTSelfAttention(config, 6)
        self.config = config
        self.hidden_act_fn = gelu

    def forward(self, hidden_states, attention_mask=None):
        hidden_states_aug = self.aug_dense(hidden_states)
        hidden_states_aug = self.attn(hidden_states_aug, attention_mask) 
        hidden_states = self.aug_dense2(hidden_states_aug)
        hidden_states = self.hidden_act_fn(hidden_states)
        return hidden_states
