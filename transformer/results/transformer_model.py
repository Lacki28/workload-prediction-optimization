import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2, 1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())

    return torch.softmax(m, -1)


def attention(Q, K, V):
    # Attention(Q, K, V) = norm(QK)V
    a = a_norm(Q, K)  # (batch_size, dim_attn, seq_length)

    return torch.matmul(a, V)  # (batch_size, seq_length, seq_length)


class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)

    def forward(self, x, kv=None):
        if (kv is None):
            # Attention with x connected to Q,K and V (For encoder)
            return attention(self.query(x), self.key(x), self.value(x))

        # Attention with x as Q, external vector kv as K an V (For decoder)
        return attention(self.query(x), self.key(kv), self.value(kv))


class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads, device):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))

        self.heads = nn.ModuleList(self.heads).to(device)

        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias=False).to(device)

    def forward(self, x, kv=None):
        a = []
        for h in self.heads:
            a.append(h(x, kv=kv))

        a = torch.stack(a, dim=-1)  # combine heads
        a = a.flatten(start_dim=2)  # flatten all head outputs

        x = self.fc(a)

        return x


class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val

        self.fc1 = nn.Linear(dim_input, dim_val, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x


class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn

        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)

    def forward(self, x):
        x = self.fc1(x)

        return x


class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn

        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)

    def forward(self, x):
        x = self.fc1(x)

        return x


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, device="cpu"):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads, device)
        self.fc1 = nn.Linear(dim_val, dim_val).to(
            device)  # Performs linear transformation on data (in features, out features)
        self.fc2 = nn.Linear(dim_val, dim_val).to(device)

        self.norm1 = nn.LayerNorm(dim_val).to(
            device)  # Applies layer normalization over a mini-batch (https://arxiv.org/abs/1607.06450) Improves computational time
        self.norm2 = nn.LayerNorm(dim_val).to(device)

    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)

        elu_tensor = F.elu(self.fc2(x))  # .to(device)

        a = self.fc1(elu_tensor)
        x = self.norm2(x + a)

        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, device="cpu"):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads, device)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads, device)
        self.fc1 = nn.Linear(dim_val, dim_val).to(device)
        self.fc2 = nn.Linear(dim_val, dim_val).to(device)

        self.norm1 = nn.LayerNorm(dim_val).to(device)
        self.norm2 = nn.LayerNorm(dim_val).to(device)
        self.norm3 = nn.LayerNorm(dim_val).to(device)

    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)

        a = self.attn2(x, kv=enc)
        x = self.norm2(a + x)

        elu_tensor = F.elu(self.fc2(x))

        a = self.fc1(elu_tensor)

        x = self.norm3(x + a)
        return x


class TimeSeriesTransformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_feat_enc, input_feat_dec, seq_len, n_decoder_layers=1,
                 n_encoder_layers=1, n_heads=1, prediction_step=1, device="cpu"):
        super(TimeSeriesTransformer, self).__init__()
        self.seq_len = seq_len

        # Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads, device))

        self.decs = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads, device))

        self.pos = PositionalEncoding(dim_val)

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_feat_enc, dim_val)
        self.dec_input_fc = nn.Linear(input_feat_dec, dim_val)
        self.out_fc = nn.Linear(self.seq_len * dim_val, prediction_step)

    def forward(self, x_enc, x_dec, training=True):
        if not training:
            for e_layer in self.encs:
                e_layer.eval()
            for d_layer in self.decs:
                d_layer.eval()
        else:
            for e_layer in self.encs:
                e_layer.train()
            for d_layer in self.decs:
                d_layer.train()
        # encoder
        first_layer = self.enc_input_fc(x_enc)
        pos_encoder = self.pos(first_layer)
        e = self.encs[0](pos_encoder)
        for enc in self.encs[1:]:
            e = enc(e)

        # decoder
        d = self.decs[0](self.dec_input_fc(x_dec), e)
        for dec in self.decs[1:]:
            d = dec(d, e)

        # output
        x = self.out_fc(d.flatten(start_dim=1))

        return x
