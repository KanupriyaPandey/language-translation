# Imports

import re
import random
import unicodedata
import numpy as np

from sklearn.model_selection import train_test_split

# Global Variables

LANGUAGE1 = 'eng'
LANGUAGE2 = 'fra'
FILEPATH = 'eng-fra.txt'

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10
PREFIXES = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re ")

#  Language Helper Class

class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

#  Data pre-processing functions

def normalize_data(input_str):
    input_str = input_str.lower()
    input_str = input_str.strip()
    input_str = unicodedata.normalize('NFD', input_str)
    input_str = ''.join(char for char in input_str if unicodedata.category(char) != 'Mn')
    input_str = re.sub(r"([.!?])", r" \1", input_str)
    input_str = re.sub(r"[^a-zA-Z!?]+", r" ", input_str)
    input_str = input_str.strip()
    return input_str

def is_valid_pair(pair):
    first_sentence_length = len(pair[0].split(' '))
    second_sentence_length = len(pair[1].split(' '))

    is_below_max_length = first_sentence_length < MAX_LENGTH and second_sentence_length < MAX_LENGTH
    starts_with_prefix = pair[1].startswith(PREFIXES)

    return is_below_max_length and starts_with_prefix

def preprocess_data(language1, language2):
    print('---Data Preprocessing---')
    lines = open(FILEPATH, encoding='utf-8').read().strip().split('\n')
    print('Number of translation pairs:', len(lines))

    line_pairs = [line.split('\t') for line in lines]
    line_pairs = [[pair[0], pair[1]] for pair in line_pairs]
    line_pairs = [[normalize_data(substring) for substring in pair] for pair in line_pairs]
    line_pairs = [list(reversed(pair)) for pair in line_pairs]

    input_language = Language(language2)
    output_language = Language(language1)

    line_pairs = [pair for pair in line_pairs if is_valid_pair(pair)]
    print('Number of translation pairs after filter:', len(line_pairs))

    for pair in line_pairs:
        input_language.add_sentence(pair[0])
        output_language.add_sentence(pair[1])

    print("Translation vocabulary:")
    print(input_language.name, input_language.n_words)
    print(output_language.name, output_language.n_words)

    return input_language, output_language, line_pairs

def split_data(line_pairs, test_size=0.2, random_state=42):
    print('\n---Split Data---')
    train_pairs, test_pairs = train_test_split(line_pairs, test_size=test_size, random_state=random_state)

    print("Number of training pairs:", len(train_pairs))
    print("Number of testing pairs:", len(test_pairs))

    return train_pairs, test_pairs


# Example usage

input_language, output_language, line_pairs = preprocess_data(LANGUAGE1, LANGUAGE2)

train_pairs, test_pairs = split_data(line_pairs, test_size=0.2)