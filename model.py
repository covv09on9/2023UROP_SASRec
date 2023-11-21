import argparse
import torch
import torch.nn as nn
from torch.utils.data import dataloader, dataset
import torch.nn.functional as F
from typing import Union, Callable
import numpy as np

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SASRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        
        if args.l2_reg > 0:
            self.regularization = nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0) 
        self.l2_reg = args.l2_reg

        self.item_emb = nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        # self._item_embedding(self.item_num+1, args.hidden_units, zero_pad=True, scale=False, with_t=False, l2_reg=args.l2_reg)

        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units) 
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  nn.MultiheadAttention(args.hidden_units,
                                                    args.num_heads,
                                                    args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
    
    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        if self.l2_reg > 0:
            reg_loss = torch.sum(self.regularization(log_seqs))
            seqs += self.l2_reg * reg_loss
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

# class Model(nn.modules):
#     device = "cuda"
#     def __init__(self,
#                  usernum,
#                  itemnum,
#                  args,
#                  is_training=True,
#                  reuse=None):
#         self.is_training = is_training
#         self.u = "hello"
#         self.input_seq = args.maxlen

    

#     def positional_encoding(dim, sentence_length, dtype=torch.float32):
#         position = torch.arange(0, sentence_length).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))

#         encoded_vec = torch.sin(position * div_term)
#         encoded_vec = torch.cat([encoded_vec, torch.cos(position * div_term)], dim=1)

#         return encoded_vec
    
#     def normalize(inputs, epsilon=1e-8):
#         inputs_shape = inputs.size()
#         params_shape = inputs_shape[-1:]

#         mean = torch.mean(inputs, dim=-1, keepdim=True)
#         variance = torch.var(inputs, dim=-1, keepdim=True)
        
#         beta = torch.nn.Parameter(torch.zeros(params_shape))
#         gamma = torch.nn.Parameter(torch.ones(params_shape))

#         normalized = (inputs - mean) / torch.sqrt(variance + epsilon)
#         outputs = gamma * normalized + beta

#         return outputs