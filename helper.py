import pandas as pd
import numpy as np
import nltk

# Sci-kit learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, f1_score
import seaborn as sns

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding

from sklearn.metrics import roc_auc_score, f1_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt


def longest_subsequence(data, character, data_column):

    """Finds the length of the longest sequence of a character in a tweet.
    INPUTS: data - dataframe containing the dataset
            character - the character whose longest sequence the function will output for each document in the dataset
            data_column - name of the column of the dataset that contains data to be processed
    OUTPUT: a list of lengths of longest sequences of characters"""

    longest_subsequences = []
    for index, row in data.iterrows():
        counter = 0
        tmp_counter = 0
        for char in row[data_column]:
            if char == character:
                tmp_counter += 1
            else:
                if tmp_counter > counter:
                    counter = tmp_counter
                tmp_counter = 0
        longest_subsequences.append(counter)
    return longest_subsequences


def prepare_embedding_matrix(vocabulary_size, tokenizer):

    """Maps the vocabulary of the dataset to English GloVe embeddings pretrained on Twitter data."""

    embeddings_index = dict()
    f = open('glove.twitter.27B.200d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocabulary_size, 200))
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    return embedding_matrix



def calculate_metrics(y_test, y_predicted):
    """
    function to calculate metrics
    """

    print(f"Accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"Precision: {precision_score(y_test, y_predicted, average='macro')}")
    print(f"Recall score: {recall_score(y_test, y_predicted, average='macro')}")
    print(f"F1 score: {f1_score(y_test, y_predicted, average='macro')}")


def plot_graph_loss(file_name, model_name):

    values = pd.read_table(file_name, sep=',')
    data = pd.DataFrame()
    data['epoch'] = list(values['epoch'].get_values() + 1) + \
                    list(values['epoch'].get_values() + 1)
    data['loss name'] = ['training'] * len(values) + \
                        ['validation'] * len(values)
    data['loss'] = list(values['loss'].get_values()) + \
                   list(values['val_loss'].get_values())
    sns.set(style='darkgrid', context='poster', font='Verdana')
    f, ax = plt.subplots()
    sns.lineplot(x='epoch', y='loss', hue='loss name', style='loss name',
                 dashes=False, data=data, palette='Set2')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend().texts[0].set_text('')
    plt.title(model_name)
    plt.show()


def confusion_matrix(y_train, y_predicted):

    cm = confusion_matrix(y_train, y_predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(cm)

def confusion_matrix_plot(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
