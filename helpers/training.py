# Imports

import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Global Variables

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10


# Function to load and prepare training data

def prepare_dataloader(input_language, output_language, train_pairs, batch_size):
    input_ids_main = np.zeros((len(train_pairs), MAX_LENGTH), dtype=np.int32)
    target_ids_main = np.zeros((len(train_pairs), MAX_LENGTH), dtype=np.int32)

    for idx, (input, target) in enumerate(train_pairs):
        input_ids = [input_language.word2index[word] for word in input.split(' ')]
        target_ids = [output_language.word2index[word] for word in target.split(' ')]

        input_ids.append(EOS_token)
        target_ids.append(EOS_token)

        input_ids_main[idx, :len(input_ids)] = input_ids
        target_ids_main[idx, :len(target_ids)] = target_ids

    input_ids_main = torch.LongTensor(input_ids_main).to(device)
    target_ids_main = torch.LongTensor(target_ids_main).to(device)

    train_data = TensorDataset(input_ids_main, target_ids_main)
    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return input_language, output_language, train_dataloader


# Function to define optimizer

def define_optimizer(encoder, decoder, learning_rate):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    return encoder_optimizer, decoder_optimizer


# Function to define loss function

def define_loss():
    criterion = nn.NLLLoss()

    return criterion

# Function to train the model

def train(train_dataloader, encoder, decoder, learning_rate, epochs=80, print_result=10):
    print('Training...')
    
    plot_losses = []
    plot_accuracies = []
    loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        correct_tokens = 0
        total_tokens = 0

        for batch_idx, data in enumerate(train_dataloader, 1):
            input_tensor, target_tensor = data

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            epoch_loss += loss.item()

            # Calculate accuracy based on non-padding tokens
            _, predicted = torch.max(decoder_outputs, 2)
            non_pad_tokens = (target_tensor != 0).sum().item()
            correct_tokens += ((predicted == target_tensor) & (target_tensor != 0)).sum().item()
            total_tokens += non_pad_tokens

        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        epoch_loss /= len(train_dataloader)

        loss_total += epoch_loss
        plot_losses.append(epoch_loss)
        plot_accuracies.append(accuracy)

        if epoch % print_result == 0:
            avg_loss = loss_total / print_result
            avg_accuracy = sum(plot_accuracies[-print_result:]) / print_result
            print(f'Epoch [{epoch}/{epochs}], Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}')
            loss_total = 0

    return plot_losses, plot_accuracies
