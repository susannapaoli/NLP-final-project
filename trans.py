import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k

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

    def forward(self, src, tgt):
        # src shape: (src_seq_len, N)
        # tgt shape: (tgt_seq_len, N)
        src_padding_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt)
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)
        src = self.dropout(self.src_embedding(src))
        tgt = self.dropout(self.tgt_embedding(tgt))
        enc_src = self.encoder(src, src_mask, src_padding_mask)
        output, _ = self.decoder(tgt, enc_src, tgt_mask, None, tgt_padding_mask, None)
        output = self.fc_out(output)
        return output

    def create_padding_mask(self, seq):
        # seq shape: (seq_len, N)
        mask = (seq == self.src_embedding.padding_idx).permute(1, 0)
        return mask

    def create_src_mask(self, src):
        # src shape: (src_seq_len, N)
        src_seq_len = src.shape[0]
        mask = torch.triu(torch.ones(src_seq_len, src_seq_len, device=self.device), diagonal=1).bool()
        return mask

    def create_tgt_mask(self, tgt):
        # tgt shape: (tgt_seq_len, N)
        tgt_seq_len = tgt.shape[0]
        mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=self.device), diagonal=1).bool()
        tgt_padding_mask = self.create_padding_mask(tgt)
        mask = mask.masked_fill(tgt_padding_mask.unsqueeze(1).unsqueeze(2), True)
        return mask
