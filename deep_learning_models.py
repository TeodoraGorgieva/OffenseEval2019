import sys
import sklearn
import tensorflow
import keras

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import matplotlib.pyplot as plt

# Sci-kit learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Reshape, Convolution2D, Convolution1D, SpatialDropout1D
from keras.layers.embeddings import Embedding
from random import shuffle

import os
import keras.backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import roc_auc_score, f1_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional, concatenate, \
                         SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Input, \
                         Flatten, GRU


from helper import prepare_embedding_matrix, longest_subsequence, calculate_metrics, confusion_matrix_plot, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


def under_sample(X, y):

    idx_0 = np.where(y == 0)[0].tolist()
    idx_1 = np.where(y == 1)[0].tolist()

    N = np.min([len(idx_0), len(idx_1)])
    idx = idx_0[:N] + idx_1[:N]
    shuffle(idx)

    X = X[idx].reshape(-1)
    y = y[idx].reshape(-1, 1)

    return X, y


def build_LSTM():

    model = Sequential()
    model.add(Embedding(vocabulary_size, 200, input_length=max_len, weights=[embedding_matrix], trainable=False))
    model.add(Dropout(0.5))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(200, dropout=0.35, recurrent_dropout=0.35))
    model.add(SpatialDropout1D(0.5))
    model.add(Dense(1, activation='sigmoid'))
    #model.add(Dense(3, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'f1_macro]'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'f1_macro]'])
    model.add(Flatten())
    model.summary()

    return model


def build_CNN(vocabulary_size, max_len, embedding_matrix):

    model = Sequential()
    model.add(Embedding(vocabulary_size, 200, input_length=max_len, weights=[embedding_matrix], trainable=False))
    model.add(Convolution1D(64, 5, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution1D(32, 3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution1D(16, 3, activation="sigmoid"))
    model.add(MaxPooling1D(5))
    model.add(Flatten())
    #model.add(Dense(X_train.shape[1], activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def build_biLSTM(EMBEDDING_DIM, vocabulary_size, max_len):

    model = Sequential()
    model.add(Embedding(vocabulary_size, 200, input_length=max_len, weights=[embedding_matrix], trainable=False))
    model.add(SpatialDropout1D(0.4))
    model.add(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)))
    model.add(Dense(1, activation='sigmoid'))
    model.add(Flatten())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def build_LSTM_CNN(EMBEDDING_DIM: object, vocabulary_size: object, max_len: object) -> object:

    model = Sequential()
    model.add(Embedding(vocabulary_size, 200, input_length=max_len, weights=[embedding_matrix], trainable=False))
    model.add(SpatialDropout1D(0.4))
    model.add(Dropout(0.4))
    #model.add(Bidirectional(CuDNNLSTM(RECURRENT_UNITS, return_sequences=True)))
    model.add(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)))
    model.add(Conv1D(64, kernel_size=2, activation='relu', padding='valid', kernel_initializer='he_uniform'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def build_GRU(EMBEDDING_DIM, vocabulary_size, max_len):

    model = Sequential()
    model.add(Embedding(vocabulary_size, 200, input_length=max_len, weights=[embedding_matrix], trainable=False))
    model.add(Bidirectional(LSTM(300, return_sequences = True, dropout=0.35, recurrent_dropout=0.35)))
    #model.add(Bidirectional(CuDNNLSTM(RECURRENT_UNITS, return_sequences=True)))
    model.add(GRU(100, dropout=0.35, recurrent_dropout=0.35))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def build_biGRU(EMBEDDING_DIM, vocabulary_size, max_len):

    model = Sequential()
    model.add(Embedding(vocabulary_size, 200, input_length=max_len, weights=[embedding_matrix], trainable=False))
    model.add(Dropout(0.4))
    model.add(Bidirectional(GRU(100, dropout=0.35, recurrent_dropout=0.35)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def build_CNN_LSTM(EMBEDDING_DIM, vocabulary_size, max_len):

    model = Sequential()
    model.add(Embedding(vocabulary_size + 1, EMBEDDING_DIM, input_length=max_len, trainable=True, name="Embeddings"))
    model.add(SpatialDropout1D(0.4))
    model.add(Conv1D(64, 4, activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Bidirectional(LSTM(100, dropout=0.0, recurrent_dropout=0.0)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def build_CNN_2D(vocab_length, max_len, X_train):

    model = Sequential()
    model.add(Embedding(vocab_length, 30, input_length=max_len))
    model.add(Reshape((30, max_len, 1)))
    model.add(Convolution2D(32, (1, 5), activation="relu"))
    model.add(Dropout(0.9))
    model.add(Convolution2D(16, (2, 3), activation="relu"))
    model.add(Dropout(0.8))
    model.add(Convolution2D(16, (2, 2), activation="relu"))
    model.add(Dropout(0.7))
    model.add(Flatten())
    #model.add(Dense(X_train.shape[1], activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def fit_model(model, X_train, y_train, split, epochs, class_weight):

    hist = model.fit(X_train,y_train,validation_split=split, epochs=epochs,class_weight=class_weight)
    return hist



if __name__ == "__main__":

    data = pd.read_csv("data/trainA.csv")

    # TOKENIZING AND CREATING SEQUENCE 1
    vocabulary_size = 30000
    max_len = 50
    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(data['tweet'])

    sequences = tokenizer.texts_to_sequences(data['tweet'])
    X = pad_sequences(sequences, maxlen=max_len)
    y = np.array(data['subtask_a'])

    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    class_weights_flag = False
    SMOTE_flag = False
    # GLOBAL VARIABLES
    EMBEDDING_DIM = 200
    class_weight = {0: 1, 1: 1}
    epochs = 10
    split = 0.2

    num_classes = 2

    if SMOTE_flag == True:
      sm = SMOTE(random_state=12)
      X_train, y_train = sm.fit_sample(X_train, y_train)

    if class_weights_flag == True:
        if num_classes == 2:
            cw = compute_class_weight("balanced", np.unique(y_train), y_train)
        elif num_classes  > 2:
            y_integers = np.argmax(y_train, axis=1)
            cw = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        class_weight = dict(enumerate(cw))


    # GLOVE WORD EMBEDDINGS
    embedding_matrix = prepare_embedding_matrix(vocabulary_size, tokenizer)

    bilstm = build_biLSTM(EMBEDDING_DIM, vocabulary_size, max_len)
    fit_model(bilstm, X_train, y_train, split, epochs, class_weight)

    #lstm_cnn_model = build_LSTM_CNN(EMBEDDING_DIM, vocabulary_size, max_len)
