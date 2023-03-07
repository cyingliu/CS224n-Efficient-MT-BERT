import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from types import SimpleNamespace

class PAL(nn.Module):
    def __init__(self, config, task_id, share_down_project=None, share_up_project=None):
        super(PAL, self).__init__()
        if isinstance(config.pal_hidden_size, list):
            pal_hidden_size = config.pal_hidden_size[task_id]
        else:
            pal_hidden_size = config.pal_hidden_size
        if not config.pal_share:
            self.down_project = nn.Linear(config.hidden_size, pal_hidden_size)
            self.up_project = nn.Linear(pal_hidden_size, config.hidden_size)
        else:
            self.down_project = share_down_project
            self.up_project = share_up_project

        attn_config = {'num_attention_heads': config.pal_attn_head,
                       'hidden_size': pal_hidden_size,
                       'attention_probs_dropout_prob': config.attention_probs_dropout_prob}
        attn_config = SimpleNamespace(**attn_config)
        self.attention = SelfAttention(attn_config)
        self.act_fn = F.gelu

    def forward(self, hidden_states, attention_mask=None):
        output = self.down_project(hidden_states)
        output = self.attention(output, attention_mask)
        output = self.up_project(output)
        output = self.act_fn(output)

        return output

class MultiPAL(nn.Module):
    def __init__(self, config):
        super(MultiPAL, self).__init__()
        if config.pal_share:
            share_down_projects = []
            share_up_projects = []
            for task_id in range(config.num_tasks):
                if isinstance(config.pal_hidden_size, list):
                    share_down_projects.append(nn.Linear(config.hidden_size, config.pal_hidden_size[task_id]))
                    share_up_projects.append(nn.Linear(config.pal_hidden_size[task_id], config.hidden_size))
                else:
                    share_down_projects.append(nn.Linear(config.hidden_size, config.pal_hidden_size))
                    share_up_projects.append(nn.Linear(config.pal_hidden_size, config.hidden_size))
            self.share_down_projects = nn.ModuleList(share_down_projects)
            self.share_up_projects = nn.ModuleList(share_up_projects)


# copy of BertSelfAttention from bert.py to avoid circular import
class SelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # this dropout is applied to normalized attention scores following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj 
    # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # attention scores are calculated by multiply query and key 
    # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number 

    # normalize the scores
    # multiply the attention scores to the value and get back V'
    # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]

    ### TODO
    bs, _, seq_len, _ = key.shape # (bs, num_attention_heads, seq_len, attention_head_size)
    attention_scores = torch.matmul(query, key.transpose(-1, -2)) # (bs, num_attention_heads, seq_len, seq_len)
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    # mask out the padding sequence
    attention_scores = attention_scores + attention_mask
    # normalize
    attention_scores = F.softmax(attention_scores, dim=-1) # (bs, num_attention_heads, seq_len, seq_len)
    # Note: hugging face bert add drop out here
    outputs = torch.matmul(attention_scores, value) # (bs, num_attention_heads, seq_len, attention_head_size)
    outputs = outputs.permute(0, 2, 1, 3)
    outputs = outputs.reshape(bs, seq_len, -1) # (bs, seq_len, num_attention_heads * attention_head_size)
    
    return outputs


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # calculate the multi-head attention 
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value

if __name__ == "__main__":
    config = {'pal': True, \
              'pal_share': False, \
              'hidden_size': 768, \
              'pal_hidden_size': 204, \
              'pal_attn_head': 12, \
              'attention_probs_dropout_prob': 0.3}
    config = SimpleNamespace(**config)
    pal = PAL(config)


