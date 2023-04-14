import random

import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        
        self.embedding = nn.Embedding(self.input_size, self.emb_size)
        
        
        if self.model_type == "RNN":
            self.rnn = nn.RNN(self.emb_size, self.encoder_hidden_size, batch_first=True)
            
        elif self.model_type == "LSTM":
            self.rnn = nn.LSTM(self.emb_size, self.encoder_hidden_size, batch_first=True)
            
        
        
        
        self.linear1 = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size)
        
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
        
        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder;
                hidden (tensor): the weights coming out of the last hidden unit
       """

        embedding = self.embedding(input)
        embedding = self.dropout(embedding)
        if self.model_type == "RNN":
            output, hidden = self.rnn(embedding)
        if self.model_type == "LSTM":
            output, (hidden, cell) = self.rnn(embedding)
            
        hidden = self.linear1(hidden)
        hidden = self.relu(hidden)
        hidden = self.linear2(hidden)
        hidden = torch.tanh(hidden)
        if self.model_type == "LSTM":
            hidden = (hidden, cell)
                                                  

        return output, hidden




class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        self.embedding = nn.Embedding(self.output_size, self.emb_size)
        if self.model_type == "RNN":
            self.rnn = nn.RNN(self.emb_size, self.decoder_hidden_size, batch_first = True)
        elif self.model_type == "LSTM":
            self.rnn = nn.LSTM(self.emb_size, self.decoder_hidden_size, batch_first = True)
        
        self.linear = nn.Linear(self.decoder_hidden_size, self.output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """

    
        
        
        N, T, hidden_dim = encoder_outputs.shape
        
        query = hidden.squeeze(0) # (N, hidden_dim)
        
        key = encoder_outputs.view(N, T, hidden_dim).transpose(1,2) # (N, hidden_dim, T)
        
        dot = torch.bmm(query.unsqueeze(1), key).squeeze(1) # (N, T)
        
        query_norm = torch.norm(query, dim=1, keepdim=True) 
        
        key_norm = torch.norm(key, dim=1) 
        
        cosine = dot / (query_norm * key_norm)
        
        
        
        attention = cosine.softmax(dim=1).unsqueeze(1) # (N, 1, T)
        
       
        return attention

    def forward(self, input, hidden, encoder_outputs=None, attention=False):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the weights coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
            where N is the batch size, T is the sequence length
        """

        embedding = self.embedding(input)
        embedding = self.dropout(embedding)
        if self.model_type == "LSTM":
          hidden, c = hidden

        if attention == True:
          weights = self.compute_attention(hidden, encoder_outputs)
          
          hidden = weights@encoder_outputs
          hidden = hidden.transpose(0,1)

          if self.model_type == "LSTM":
            weights_c = self.compute_attention(c, encoder_outputs)
            c = weights_c@encoder_outputs
            c = c.transpose(0,1)
        
        
        if self.model_type == "LSTM":
          hidden = (hidden, c)
        
        
        output, hidden = self.rnn(embedding,hidden)
        
        
        output = self.linear(output.squeeze(1))
        output = self.log_softmax(output)
        
        
        return output, hidden



class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device, attention=False):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.attention=attention  #if True attention is implemented
      
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
       

    def forward(self, source, out_seq_len=None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        batch_size = source.shape[0]
        if out_seq_len is None:
            out_seq_len = source.shape[1]
        
     
        outputs = []
        encoder_output, encoder_hidden = self.encoder(source)
        
        decoder_input = source[:, 0].unsqueeze(1)
        for t in range(out_seq_len):
          output, hidden = self.decoder(decoder_input, encoder_hidden, encoder_outputs=encoder_output, attention=self.attention)
          
          
          outputs.append(output)
          decoder_input = output.argmax(dim=-1).unsqueeze(1)
  
          encoder_hidden = hidden
        outputs = torch.stack(outputs).transpose(0,1)
        
        return outputs