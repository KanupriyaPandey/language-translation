
from __future__ import unicode_literals, print_function, division

from flask import Flask, render_template, request

import torch
# from your_model_file import EncoderRNN, AttnDecoderRNN, Lang  # Import your model classes and Lang

from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import numpy as np

import pickle

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
from torchtext.data.metrics import bleu_score


app = Flask(__name__)

EOS_token = 1

# Helper functions

from helpers.preprocessing import preprocess_data, split_data, normalize_data
from helpers.model import Encoder, AttentionDecoder
from helpers.training import prepare_dataloader, train
from helpers.evaluation import generate_translation, evaluate

# Global Variables

languages = ['eng', 'fra', 'spa', 'deu', 'por']

LANGUAGE1 = 'eng'
LANGUAGE2 = 'fra'
FILEPATH1 = f'data/{LANGUAGE1}-{LANGUAGE2}.txt'

LANGUAGE3 = 'eng'
LANGUAGE4 = 'spa'
FILEPATH2 = f'data/{LANGUAGE3}-{LANGUAGE4}.txt'

LANGUAGE5 = 'eng'
LANGUAGE6 = 'deu'
FILEPATH3 = f'data/{LANGUAGE5}-{LANGUAGE6}.txt'

LANGUAGE7 = 'eng'
LANGUAGE8 = 'por'
FILEPATH4 = f'data/{LANGUAGE7}-{LANGUAGE8}.txt'

# Model tuning parameters

HIDDEN_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT = 0.1


"""Data preprocessing"""

input_language1, output_language1, line_pairs1 = preprocess_data(FILEPATH1, LANGUAGE1, LANGUAGE2)
input_language2, output_language2, line_pairs2 = preprocess_data(FILEPATH2, LANGUAGE3, LANGUAGE4)
input_language3, output_language3, line_pairs3 = preprocess_data(FILEPATH3, LANGUAGE5, LANGUAGE6)
input_language4, output_language4, line_pairs4 = preprocess_data(FILEPATH4, LANGUAGE7, LANGUAGE8)


"""Model"""
#Fra-Eng
encoder_fra_eng_model = Encoder(input_language1.n_words, HIDDEN_SIZE, DROPOUT).to(device)
decoder_fra_eng_model = AttentionDecoder(HIDDEN_SIZE, output_language1.n_words, DROPOUT).to(device)
# encoder_eng_fra_model, decoder_eng_fra_model

#Spa-Eng
encoder_spa_eng_model = Encoder(input_language2.n_words, HIDDEN_SIZE, DROPOUT).to(device)
decoder_spa_eng_model = AttentionDecoder(HIDDEN_SIZE, output_language2.n_words, DROPOUT).to(device)

#Deu-Eng
encoder_deu_eng_model = Encoder(input_language3.n_words, HIDDEN_SIZE, DROPOUT).to(device)
decoder_deu_eng_model = AttentionDecoder(HIDDEN_SIZE, output_language3.n_words, DROPOUT).to(device)

#Por-Eng
encoder_por_eng_model = Encoder(input_language4.n_words, HIDDEN_SIZE, DROPOUT).to(device)
decoder_por_eng_model = AttentionDecoder(HIDDEN_SIZE, output_language4.n_words, DROPOUT).to(device)


# Load models
encoder_fra_eng = f'saved_models/encoder-eng-fra-mx15_withoutprefx.pth'
decoder_fra_eng = f'saved_models/decoder-eng-fra-mx15_withoutprefx.pth'

encoder_spa_eng = f'saved_models/encoder-eng-spa-mx15_withoutprefx.pth'
decoder_spa_eng = f'saved_models/decoder-eng-spa-mx15_withoutprefx.pth'

encoder_deu_eng = f'saved_models/encoder-eng-deu-mx15_withoutprefx.pth'
decoder_deu_eng = f'saved_models/decoder-eng-deu-mx15_withoutprefx.pth'

encoder_por_eng = f'saved_models/encoder-eng-por-mx15_withoutprefx.pth'
decoder_por_eng = f'saved_models/decoder-eng-por-mx15_withoutprefx.pth'


encoder_fra_eng_model.load_state_dict(torch.load(encoder_fra_eng, map_location=torch.device('cpu')))
decoder_fra_eng_model.load_state_dict(torch.load(decoder_fra_eng, map_location=torch.device('cpu')))


encoder_spa_eng_model.load_state_dict(torch.load(encoder_spa_eng, map_location=torch.device('cpu')))
decoder_spa_eng_model.load_state_dict(torch.load(decoder_spa_eng, map_location=torch.device('cpu')))

encoder_deu_eng_model.load_state_dict(torch.load(encoder_deu_eng, map_location=torch.device('cpu')))
decoder_deu_eng_model.load_state_dict(torch.load(decoder_deu_eng, map_location=torch.device('cpu')))


encoder_por_eng_model.load_state_dict(torch.load(encoder_por_eng, map_location=torch.device('cpu')))
decoder_por_eng_model.load_state_dict(torch.load(decoder_por_eng, map_location=torch.device('cpu')))


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def translate_text(input_text, encoder, decoder, input_lang, output_lang):
    with torch.no_grad():
        unknown_words = []

        # Check for unknown words in the input sentence
        for word in input_text.split():
            if word not in input_lang.word2index:
                unknown_words.append(word)

        if unknown_words:
            return f"Words not found in vocabulary: {', '.join(unknown_words)}"

        input_tensor = tensorFromSentence(input_lang, input_text)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                # decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return ' '.join(decoded_words)


def translate_text_fra_eng(input_text):
    return translate_text(input_text, encoder_fra_eng_model, decoder_fra_eng_model, input_language1, output_language1)

def translate_text_spa_eng(input_text):
    return translate_text(input_text, encoder_spa_eng_model, decoder_spa_eng_model, input_language2, output_language2)


def translate_text_deu_eng(input_text):
    return translate_text(input_text, encoder_deu_eng_model, decoder_deu_eng_model, input_language3, output_language3)


def translate_text_por_eng(input_text):
    return translate_text(input_text, encoder_por_eng_model, decoder_por_eng_model, input_language4, output_language4)


# Other routes and functions remain the same

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        input_text = request.form['input_text']
        selected_language = request.form['language']

        # Preprocess the input text before translation
        normalized_text = unicodedata.normalize('NFD', input_text)
        preprocessed_text = re.sub(r'[\u0300-\u036f]', '', normalized_text)
        preprocessed_text = re.sub(r'[^a-zA-Z!?]+', ' ', preprocessed_text)
        preprocessed_text = preprocessed_text.strip()

        # Convert input text to lowercase
        preprocessed_text = preprocessed_text.lower() 

        # Initialize the translated_text variable
        translated_text = ""

        # Perform translation only if input text is present
        if preprocessed_text:
            # Perform translation based on the selected language
            if selected_language == 'eng_fra':
                translated_text = translate_text_fra_eng(preprocessed_text)
            elif selected_language == 'eng_por':
                translated_text = translate_text_por_eng(preprocessed_text)
            elif selected_language == 'eng_spa':
                translated_text = translate_text_spa_eng(preprocessed_text)
            elif selected_language == 'eng_due':
                translated_text = translate_text_deu_eng(preprocessed_text)
            else:
                translated_text = "Unsupported language"

        # Render the template based on input and translated text
        return render_template('index.html', input_text=input_text, translated_text=translated_text)

    # For GET requests or when there's no input text, render the template with empty translated_text
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
 

