# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:56:08 2017

@author: nhj
"""

import nltk
from __future__ import absolute_import, division, print_function

%matplotlib inline
# %matplotlib nbagg
import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from data_generator_tensorflow import get_batch, print_valid_characters

import os
import sys
sys.path.append(os.path.join('.', '..')) 
import utils 

import tf_utils

import string


DATA_DIR = "C:/Users/nhj/Desktop/playground/Author Autocomplete/austen.txt"

with open(DATA_DIR, 'r') as file:
    data = file.read()

#tokenize and clean data
tokenized_text = nltk.word_tokenize(data.lower())
data = tokenized_text

# import word features from FastText
# and map words to word features

N_WORD_FEATURES = 200
SEQ_LENGTH = 4


X = np.zeros((len(data)/SEQ_LENGTH, SEQ_LENGTH, N_WORD_FEATURES))
y = np.zeros((len(data)/SEQ_LENGTH, SEQ_LENGTH, N_WORD_FEATURES))
for i in range(0, len(data)/SEQ_LENGTH):
    X_sequence = data[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
    X_sequence_ix = [char_to_ix[value] for value in X_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, N_WORD_FEATURES))
    for j in range(SEQ_LENGTH):
        input_sequence[j][X_sequence_ix[j]] = 1.
    X[i] = input_sequence

    y_sequence = data[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]
    y_sequence_ix = [char_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, N_WORD_FEATURES))
    for j in range(SEQ_LENGTH):
        target_sequence[j][y_sequence_ix[j]] = 1.
    y[i] = target_sequence

# Define validation data


# Set up hyperparameters
# Build the input layer(s)
# Build the hidden layer(s)
# Define the output layer


HIDDEN_DIM = 100
SAVE_FOLDER = "C:/Users/nhj/Desktop/playground/Author AutoComplete/"

model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, N_WORD_FEATURES), return_sequences=True))
model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(N_WORD_FEATURES)))
model.add(Activation('softmax'))
# Define the loss and validation function
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")



# (Optional) Test the forward pass

# Train model

nb_epoch = 0
while True:
    print('\n\n')
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
    nb_epoch += 1
    generate_text(model, GENERATE_LENGTH)
    if nb_epoch % 10 == 0:
        model.save_weights(SAVE_FOLDER + 'checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, nb_epoch))



# Manually test results of model

def generate_text(model, length):
    ix = [np.random.randint(N_WORD_FEATURES)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, N_WORD_FEATURES))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)


# Plot results of model


# Save model






