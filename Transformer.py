import numpy as np
import torch
from torch import nn
import random

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, pad_idx, N_layers, batch, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
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
        super(TransformerTranslator, self).__init__()
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
        self.pad_idx = pad_idx
        self.n_layers = N_layers
        
        seed_torch(0)
        
#Initialize the embedding layer and the positional embedding #
        self.embeddingL = nn.Embedding(self.input_size,self.hidden_dim)
        self.posembeddingL = nn.Embedding(self.max_length,self.hidden_dim)
# Initialize multi-head self-attention for as many heads as we want #
        self.heads = {}
        for head in range(self.num_heads):
          k = nn.Linear(self.hidden_dim, self.dim_k).to(self.device)
          v = nn.Linear(self.hidden_dim, self.dim_v).to(self.device)
          q = nn.Linear(self.hidden_dim, self.dim_q).to(self.device)
        
          self.heads[head] = (q,k,v)
        self.softmax = nn.Softmax(dim=-1)
        
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)
# Initialize  the feed-forward layer and its normalization # 

        self.ff_lay1 = nn.Linear(self.hidden_dim,self.dim_feedforward) 
        self.relu = nn.ReLU()
        self.ff_lay2 = nn.Linear(self.dim_feedforward,self.hidden_dim) 
        self.ff_normalize = nn.LayerNorm(self.hidden_dim)
 # Initialize  the final layer   #
        self.lin_out = nn.Linear(self.hidden_dim,self.output_size)
        self.softm = nn.Softmax(dim = -1)

    def forward(self, inputs, target):
        """
        This function computes the full Transformer forward pass.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.
               targets: a PyTorch tensor containing golden labels

        :returns: the model outputs. 
        """
# Implemented  full Transformer stack for the forward pass looping over heads and layers and 
# using cross-self attention on decoder inputs and encoder outputs after the masked attention 
# on decoder embeddings.
    
        embeddings_enc = self.embed(inputs)
        embeddings_dec = self.embed(target)
        for i in range(self.n_layers):
            
            attention_enc = self.multi_head_attention(embeddings_enc)
            encoder_output = self.feedforward_layer(attention_enc)
            embeddings_enc = encoder_output
        
        
        for i in range(self.n_layers):
            attention_decoder = self.multi_head_attention_mask(embeddings_dec)
            attention_dec = self.decoder_attention(attention_decoder,encoder_output)
            decoder_output = self.feedforward_layer(attention_dec)
            embeddings_dec = decoder_output
            
        outputs = self.final_layer(decoder_output)
        
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        #############################################################################
        # Embedding layer combining normal and positional embedding                 #
        #############################################################################
        token_emb = self.embeddingL(inputs)
        positional_emb = self.posembeddingL(torch.arange(inputs.shape[1]).to(self.device))
        embeddings = torch.add(token_emb, positional_emb)

        return embeddings
        
    def multi_head_attention_mask(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        We include a padding mask here, so that pads are ignored. In this way we implement
        a complete version of the transformer architecture
        """
        #############################################################################
        #        MASKED multi-head self-attention followed by add + norm.           #
        #############################################################################
        attentions = []
 
        for vectors in self.heads.values():
          q = vectors[0](inputs)
          k = vectors[1](inputs).to(self.device)
          mask_k = (k != self.pad_idx)
          mask_k = mask_k * -1e9
          mask_q = (q != self.pad_idx)
          mask_q = mask_q * -1e9
          v = vectors[2](inputs).to(self.device)
            
          mask_s = (mask_q @ mask_k.transpose(-2, -1))
          s = (q @ k.transpose(-2,-1))/np.sqrt(self.dim_k)
          
 
          scores = s + mask_s
          s = self.softmax(scores).to(self.device)
          
          att = s @ v
          attentions.append(att)        
        multi_head = torch.cat([heads for heads in attentions],dim = -1)

        outputs = self.attention_head_projection(multi_head)
        outputs = self.norm_mh(torch.add(inputs,outputs))
        
        return outputs
    
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        This version does not include a padding mask. This is a simplified implementation
        used as a first self-attention layer on both encoder and decoder's embeddings.
        """       
        
        ################################################################################
        #                                                                              #
        #       Multi-head self-attention for the encoder followed by add + norm.      #
        #                                                                              #
        ################################################################################
        attentions = []
        for vectors in self.heads.values():
          q = vectors[0](inputs)
          k = vectors[1](inputs).to(self.device)
          v = vectors[2](inputs).to(self.device)
          s = self.softmax((q @ k.transpose(-2,-1))/np.sqrt(self.dim_k)).to(self.device)
          att = s @ v
          attentions.append(att)        
        multi_head = torch.cat([heads for heads in attentions],dim = -1)

        outputs = self.attention_head_projection(multi_head)
        outputs = self.norm_mh(torch.add(inputs,outputs))
        
        return outputs
    
    def decoder_attention(self, inputs, encoder_output):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
               encoder_outputs: Tensor including the outputs of the encoder 
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        This version of the attention reproduces the cross self-attention mechanism.
        It matches the embedded and masked decoder input(using teaching enforcement) and
        encoder's output to better understand the order of the words in the translations.
        """       
        
        ################################################################################
        #                                                                              #
        # Implemented multi-head self-attention for the decoder followed by add + norm.#
        # This does cross attention combining encoder output and decoder input         #
        ################################################################################
        attentions = []
        for vectors in self.heads.values():
          q = vectors[0](inputs)
          k = vectors[1](encoder_output).to(self.device)
          v = vectors[2](encoder_output).to(self.device)
          s = self.softmax((q @ k.transpose(-2,-1))/np.sqrt(self.dim_k)).to(self.device)
          att = s @ v
          attentions.append(att)        
        multi_head = torch.cat([heads for heads in attentions],dim = -1)

        outputs = self.attention_head_projection(multi_head)
        outputs = self.norm_mh(torch.add(inputs,outputs))
       
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        #                                                                           #
        #  Implemented 2-layer feedforward using ReLU activations followed          #
        #   by add + norm.                                                          #
        #############################################################################
        lay1 = self.ff_lay1(inputs)
        lay2 = self.ff_lay2(self.relu(lay1))

        outputs = self.ff_normalize(torch.add(lay2,inputs))

        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        #############################################################################
        # Implement the final layer for the Transformer Translator.                 #
        #############################################################################
        outputs = self.lin_out(inputs)

        return outputs
        

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
