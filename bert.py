from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from pal import PAL, MultiPAL
from prefix import Prefix
from houlsby import Houlsby
from utils import *


class BertSelfAttention(nn.Module):
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


class BertLayer(nn.Module):
  def __init__(self, config, pal_multilayers=None, layer_id=0):
    super().__init__()
    # multi-head attention
    self.self_attention = BertSelfAttention(config)
    # add-norm
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # another add-norm
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
    
    self.layer_id = layer_id
    ##### Adaptatation Modules #####
    self.use_pal = config.pal
    self.use_prefix = config.prefix
    self.get_init_prefix = config.prefix_get_init
    self.use_houlsby = config.houlsby
    # PAL
    if config.pal:
      pals = []
      for i in range(config.num_tasks):
        if config.pal_share:
          pal = PAL(config, \
                    task_id=i,
                    share_down_project=pal_multilayers.share_down_projects[i], \
                    share_up_project=pal_multilayers.share_up_projects[i])
        else:
          pal = PAL(config, task_id=i)
        pals.append(pal)
      self.pals = nn.ModuleList(pals)
    
    # Prefix
    if config.prefix:
      self.prefixs = nn.ModuleList([Prefix(config, layer_id=layer_id, task_id=task_id) for task_id in range(config.num_tasks)])
    
    # Houlsby
    if config.houlsby:
      self.houlsbys_0 = nn.ModuleList([Houlsby(config, task_id=i) for i in range(config.num_tasks)])
      self.houlsbys_1 = nn.ModuleList([Houlsby(config, task_id=i) for i in range(config.num_tasks)])
      self.houlsby_add_layernorm = config.houlsby_add_layernorm
    ################################



  def add_norm(self, inputs, outputs, dense_layer, dropout, ln_layer, task_id=0, position=0):
    """
    this function is applied after the multi-head attention layer or the feed forward layer
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Hint: Remember that BERT applies to the output of each sub-layer, before it is added to the sub-layer input and normalized 
    ### TODO
    outputs = dense_layer(outputs)
    outputs = dropout(outputs)

    ##### Adaptation Modules ######
    if self.use_houlsby:
      if position == 0:
        outputs = self.houlsbys_0[task_id](outputs)
      else:
        outputs = self.houlsbys_1[task_id](outputs)

    if self.use_houlsby and self.houlsby_add_layernorm:
      if position == 0:
        outputs = self.houlsbys_0[task_id].layernorm(outputs + inputs)
      else:
        outputs = self.houlsbys_1[task_id].layernorm(outputs + inputs)
    else: # default or houlsby without add layernorm
      outputs = ln_layer(outputs + inputs)
    ###############################

    

    return outputs


  def forward(self, hidden_states, attention_mask, task_id=0):
    """
    hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
    each block consists of 
    1. a multi-head attention layer (BertSelfAttention)
    2. a add-norm that takes the input and output of the multi-head attention layer
    3. a feed forward layer
    4. a add-norm that takes the input and output of the feed forward layer
    """
    ### TODO
    ##### Adaptation Modules #####
    if self.get_init_prefix:
      init_emb = torch.mean(hidden_states, dim=(0, 1)).detach().cpu()
      if not os.path.exists('weights'):
          os.mkdir('weights')
      torch.save(init_emb, os.path.join('weights', f'emb_{self.layer_id}.pt'))
      print(f"Saving weights to {os.path.join('weights', f'emb_{self.layer_id}.pt')}")
    
    if self.use_prefix:
      hidden_states, attention_mask = self.prefixs[task_id](hidden_states, attention_mask)
    ##############################

    attention_outputs = self.self_attention(hidden_states, attention_mask)
    norm_outputs = self.add_norm(hidden_states, attention_outputs, self.attention_dense, self.attention_dropout, self.attention_layer_norm, task_id=task_id, position=0)
    interm_outputs = self.interm_af(self.interm_dense(norm_outputs))
    
    ##### Adaptation Modules #####
    if self.use_pal:
      extra = self.pals[task_id](hidden_states, attention_mask)
      norm_outputs = norm_outputs + extra
    ##############################

    outputs = self.add_norm(norm_outputs, interm_outputs, self.out_dense, self.out_dropout, self.out_layer_norm, task_id=task_id, position=1)

    return outputs



class BertModel(BertPreTrainedModel):
  """
  the bert model returns the final embeddings for each token in a sentence
  it consists
  1. embedding (used in self.embed)
  2. a stack of n bert layers (used in self.encode)
  3. a linear transformation layer for [CLS] token (used in self.forward, as given)
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # embedding
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # position_ids (1, len position emb) is a constant, register to buffer
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    ##### Adaptation Modules #####
    # PAL
    if config.pal:
      pal_multilayers = MultiPAL(config)
      self.bert_layers = nn.ModuleList([BertLayer(config, pal_multilayers, layer_id=i) for i in range(config.num_hidden_layers)])
    else: # default (no adaptation), prefix
      # bert encoder
      self.bert_layers = nn.ModuleList([BertLayer(config, layer_id=i) for i in range(config.num_hidden_layers)])
    if config.prefix:
      self.cls_pos = {}
      for i in range(config.num_tasks):
        if isinstance(config.prefix_length, list):
          self.cls_pos[i] = config.prefix_length[i]
        else:
          self.cls_pos[i] = config.prefix_length
    else:
      self.cls_pos = {0: 0, 1: 0, 2: 0}
    ##############################

    # for [CLS] token
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # Get word embedding from self.word_embedding into input_embeds.
    inputs_embeds = self.word_embedding(input_ids) # (bs, seq_length, hidden_size)


    # Get position index and position embedding from self.pos_embedding into pos_embeds.
    pos_ids = self.position_ids[:, :seq_length] # (1, seq_length)
    pos_embeds = self.pos_embedding(pos_ids) # (1, seq_length, hidden_size)

    # Get token type ids, since we are not consider token type, just a placeholder.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # Add three embeddings together; then apply embed_layer_norm and dropout and return.   
    embeds = inputs_embeds + pos_embeds + tk_type_embeds # (bs, seq_length, hidden_size)
    embeds = self.embed_layer_norm(embeds)
    embeds = self.embed_dropout(embeds)

    return embeds


  def encode(self, hidden_states, attention_mask, task_id=0):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # get the extended attention mask for self attention
    # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
    # non-padding tokens with 0 and padding tokens with a large negative number 
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # pass the hidden states through the encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      # feed the encoding from the last bert_layer to the next
      hidden_states = layer_module(hidden_states, extended_attention_mask, task_id=task_id)

    return hidden_states

  def forward(self, input_ids, attention_mask, task_id=0):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask, task_id=task_id)

    # get cls token hidden state
    first_tk = sequence_output[:, self.cls_pos[task_id]]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
