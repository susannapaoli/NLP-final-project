import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hid_dim, n_heads, pf_dim, dropout),
            n_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hid_dim, n_heads, pf_dim, dropout),
            n_layers
        )
        self.src_embedding = nn.Embedding(input_dim, hid_dim)
        self.tgt_embedding = nn.Embedding(output_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        src_len, N = src.shape
        tgt_len, N = tgt.shape
        src = self.dropout(self.src_embedding(src))
        tgt = self.dropout(self.tgt_embedding(tgt))
        enc_src = self.encoder(src, src_mask, src_padding_mask)
        output, _ = self.decoder(tgt, enc_src, tgt_mask, None, tgt_padding_mask, None)
        output = self.fc_out(output)
        return output
