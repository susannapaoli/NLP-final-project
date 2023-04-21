import numpy as np
import torch
from torch import nn
import random

class TransformerTranslatorBaseline(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary = vocab size
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslatorBaseline, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
# Initialized embedding lookup#
        self.embeddingL = nn.Embedding(self.input_size,self.hidden_dim)
        self.posembeddingL = nn.Embedding(self.max_length,self.hidden_dim)
# Initializations for 2 self-attention heads #        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)

        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)
        
# Initialize what you need for the feed-forward layer and its normalization# 
        self.ff_lay1 = nn.Linear(self.hidden_dim,self.dim_feedforward) 
        self.relu = nn.ReLU()
        self.ff_lay2 = nn.Linear(self.dim_feedforward,self.hidden_dim) 
        self.ff_normalize = nn.LayerNorm(self.hidden_dim)
# Initialize what you need for the final layer 
        self.lin_out = nn.Linear(self.hidden_dim,self.output_size)
        self.softm = nn.Softmax(dim = 2)

    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        It includes a single layer transformer decoder with a simplified self-attention

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """
# Implement the full Transformer stack for the forward pass. 
        embeddings = self.embed(inputs)
        attention = self.multi_head_attention(embeddings)
        ff = self.feedforward_layer(attention)
        outputs = self.final_layer(ff)

        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
# Implement the embedding lookup by adding positional embeddings to the classic ones

        token_emb = self.embeddingL(inputs)
        positional_emb = self.posembeddingL(torch.arange(inputs.shape[1]).to(self.device))
        embeddings = torch.add(token_emb,positional_emb)  

        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
# Implemented simplified multi-head self-attention followed by add + norm.

        K1 = self.k1(inputs)
        V1 = self.v1(inputs)
        Q1 = self.q1(inputs)
        QK1 = self.softmax(Q1 @ K1.transpose(-2, -1) / np.sqrt(self.dim_k))
        att1 = QK1 @ V1
        K2 = self.k2(inputs)
        V2 = self.v2(inputs)
        Q2 = self.q2(inputs)
        QK2 = self.softmax(Q2 @ K2.transpose(-2, -1) / np.sqrt(self.dim_k))
        att2 = QK2 @ V2
        head = torch.cat((att1, att2), dim = 2)
        outputs = self.attention_head_projection(head)
        outputs = self.norm_mh(torch.add(outputs, inputs))
        
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
# Implement the feedforward layer, with ReLU activations followed by add + norm.    #
        lay1 = self.ff_lay1(inputs)
        lay2 = self.ff_lay2(self.relu(lay1))

        outputs = self.ff_normalize(torch.add(lay2,inputs))

        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
# Implemented  final predictive layer for the Transformer Translator.  #
        
        outputs = self.lin_out(inputs)

        return outputs
        

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
