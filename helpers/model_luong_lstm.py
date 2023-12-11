# Imports

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Global Variables

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10


# Encoder class

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder_layers = nn.Sequential(
            nn.Embedding(input_size, hidden_size),
            nn.Dropout(dropout_p),
            nn.GRU(hidden_size, hidden_size, batch_first=True)
        )

    def forward(self, input):
        output, hidden = self.encoder_layers(input)
        return output, hidden

# TODO: refractor decoder classes

# Decoder class 

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

# Decoder with attention class


# Decoder with LSTM and Luong attention class

class LuongAttentionLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(LuongAttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)

    # print('Before torch.bmm: encoder_outputs', encoder_outputs)
    def forward(self, decoder_hidden, encoder_outputs):
        
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.squeeze(0).unsqueeze(1).expand(-1, seq_len, -1)
        
        print('Before torch.bmm: encoder_outputs shape', encoder_outputs.shape)
        print('Before torch.bmm: decoder_hidden shape', decoder_hidden.shape)

        energy = torch.bmm(encoder_outputs, decoder_hidden.transpose(1, 2))
        print('After torch.bmm: encoder_outputs', encoder_outputs.shape)
        print('After torch.bmm: decoder_hidden', decoder_hidden.shape)

        attn_weights = F.softmax(energy, dim=1)
        context = torch.bmm(attn_weights, encoder_outputs)
        
        return context, attn_weights



class AttentionDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p):

        super(AttentionDecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = LuongAttentionLSTM(hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)  # LSTM instead of GRU
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_size = hidden_size  # Add this line to store hidden_size as an instance attribute

    
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):

        # print('LSTM luong: encoder_hidden ', encoder_hidden)
        # print('LSTM luong encoder_hidden shape: ', encoder_hidden.shape)
        # print('LSTM luong encoder_hidden type: ', type(encoder_hidden))
        # LSTM hidden state tuple initialization
      
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, hidden, attentions

   

    # def forward_step(self, input, hidden, encoder_outputs):
    #     embedded = self.dropout(self.embedding(input))

    #     context, attn_weights = self.attention(hidden, encoder_outputs)
    #     input_lstm = torch.cat((embedded, context), dim=2)

    #     output, hidden = self.lstm(input_lstm, hidden)
    #     output = self.out(output)

    #     return output, hidden, attn_weights
    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        context, attn_weights = self.attention(hidden, encoder_outputs)
        
        # Print shapes for debugging
        print("Embedded shape:", embedded.shape)
        print("Context shape:", context.shape)

        # Ensure compatibility in dimensions except dimension 1 for concatenation
        if embedded.size(0) != context.size(0):
            # Handle size mismatch in dimension 1
            # Add your handling logic here based on the data and desired outcome
            pass
        else:
            # Expand context to match the sequence length of embedded tensor
            context = context.unsqueeze(1).repeat(1, embedded.size(1), 1)

            input_lstm = torch.cat((embedded, context), dim=2)

            output, hidden = self.lstm(input_lstm, hidden)
            output = self.out(output)

            return output, hidden, attn_weights















