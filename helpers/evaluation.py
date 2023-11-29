# Imports

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Global Variables

SOS_token = 0
EOS_token = 1


# Function to generate translation to evaluate the model

def generate_translation(input_language, output_language, sentence, encoder, decoder):
    with torch.no_grad():

        indexes = [input_language.word2index[word] for word in sentence.split(' ')]
        indexes.append(EOS_token)
        input_tensor = torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_language.index2word[idx.item()])

    return decoded_words, decoder_attn


# Function to evaluate and print translation samples

def evaluate(input_language, output_language, encoder, decoder, pairs, evaluate_train=True, iterations=10):
    candidate_corpus = []
    references_corpus = []

    for i in range(iterations):
        if evaluate_train:
            pair = random.choice(pairs)
        else:
            pair = pairs[i]
        
        input_seq = pair[0]
        print('Input Sequence>', input_seq)

        target_seq = pair[1].split()
        print('Target Sequence =', target_seq)

        pred_seq, _ = generate_translation(input_language, output_language, input_seq, encoder, decoder)
        pred_seq = pred_seq[:-1]
        print('Predicted sequence <', pred_seq)

        candidate_corpus.append(pred_seq)
        references_corpus.append([target_seq])

    return candidate_corpus, references_corpus
