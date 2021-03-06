from random import shuffle

import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Bidirectional, SpatialDropout1D, Flatten
# Keras
from keras.layers import Reshape, Convolution2D, Convolution1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
# Sci-kit learn
from sklearn.model_selection import train_test_split

from helper import prepare_embedding_matrix, calculate_metrics


def under_sample(X, y):
    idx_0 = np.where(y == 0)[0].tolist()
    idx_1 = np.where(y == 1)[0].tolist()

    N = np.min([len(idx_0), len(idx_1)])
    idx = idx_0[:N] + idx_1[:N]
    shuffle(idx)

    X = X[idx].reshape(-1)
    y = y[idx].reshape(-1, 1)

    return X, y


def build_LSTM(vocabulary_size, max_len, embedding_matrix):
    model_glove = Sequential()
    model_glove.add(Embedding(vocabulary_size, 200, input_length=max_len, weights=[embedding_matrix], trainable=False))
    model_glove.add(Dropout(0.5))
    model_glove.add(Conv1D(64, 5, activation='relu'))
    model_glove.add(MaxPooling1D(pool_size=4))
    model_glove.add(LSTM(200))
    # For sub-classes A and B: activation function='sigmoid' and loss='binary_crossentropy'
    # For sub-class C: activation function='softmax' and loss='categorical_crossentropy'
    model_glove.add(Dense(1, activation='softmax'))
    model_glove.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model_glove


def build_CNN(vocab_length, max_len, X_train):
    model = Sequential()
    model.add(Embedding(vocab_length, 30, input_length=max_len))
    model.add(Convolution1D(64, 5, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution1D(32, 3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution1D(16, 3, activation="sigmoid"))
    model.add(MaxPooling1D(5))
    model.add(Flatten())
    # model.add(Dense(X_train.shape[1], activation='softmax'))
    # For sub-classes A and B: activation function='sigmoid' and loss='binary_crossentropy'
    # For sub-class C: activation function='softmax' and loss='categorical_crossentropy'
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def build_LSTM_CNN(EMBEDDING_DIM, vocabulary_size, max_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size + 1, EMBEDDING_DIM, input_length=max_len, trainable=True, name="Embeddings"))
    model.add(SpatialDropout1D(0.4))
    model.add(Dropout(0.4))
    # model.add(Bidirectional(CuDNNLSTM(RECURRENT_UNITS, return_sequences=True)))
    model.add(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)))
    model.add(Conv1D(64, kernel_size=2, activation='relu', padding='valid', kernel_initializer='he_uniform'))
    model.add(Flatten())
    # For sub-classes A and B: activation function='sigmoid' and loss='binary_crossentropy'
    # For sub-class C: activation function='softmax' and loss='categorical_crossentropy'
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def build_CNN_LSTM(EMBEDDING_DIM, vocabulary_size, max_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size + 1, EMBEDDING_DIM, input_length=max_len, trainable=True, name="Embeddings"))
    model.add(SpatialDropout1D(0.4))
    model.add(Conv1D(64, 4, activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Bidirectional(LSTM(100, dropout=0.0, recurrent_dropout=0.0)))
    # For sub-classes A and B: activation function='sigmoid' and loss='binary_crossentropy'
    # For sub-class C: activation function='softmax' and loss='categorical_crossentropy'
    model.add(Dense(1, activation='softmax'))
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
    # model.add(Dense(X_train.shape[1], activation='softmax'))
    # For sub-classes A and B: activation function='sigmoid' and loss='binary_crossentropy'
    # For sub-class C: activation function='softmax' and loss='categorical_crossentropy'
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


if __name__ == "__main__":
    data = pd.read_csv("data/trainC.csv")

    # TOKENIZING AND CREATING SEQUENCE 1
    vocabulary_size = 30000
    max_len = 100
    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(data['tweet'])

    sequences = tokenizer.texts_to_sequences(data['tweet'])
    X = pad_sequences(sequences, maxlen=max_len)
    y = np.array(data['subtask_c'])

    y = y.reshape(-1, 1)

    # GLOBAL VARIABLES
    EMBEDDING_DIM = 200
    class_weight = {0: 1, 1: 1, 2: 1}
    epochs = 10
    split = 0.2

    # GLOVE WORD EMBEDDINGS
    embedding_matrix = prepare_embedding_matrix(vocabulary_size, tokenizer)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=0, stratify=y)

    lstm_model = build_LSTM(vocabulary_size, max_len, embedding_matrix)

    # cnn_model = build_CNN(vocabulary_size, max_len, X_train)

    # cnn_model2 = build_CNN_2D(vocabulary_size, max_len, X_train)

    # cnn_lstm_model = build_CNN_LSTM(EMBEDDING_DIM, vocabulary_size, max_len)

    # lstm_cnn_model = build_LSTM_CNN(EMBEDDING_DIM, vocabulary_size, max_len)

    hist = lstm_model.fit(X_train,
                          y_train,
                          validation_split=split,
                          epochs=epochs,
                          class_weight=class_weight)

    y_predicted = lstm_model.predict(X_test)
    # y_predicted = np.where(y_predicted > 0.5, 1, 0)
    calculate_metrics(y_test, y_predicted)
