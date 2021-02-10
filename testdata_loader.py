import pandas as pd

import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import emoji
import num2words

from nltk import pos_tag
from nltk.corpus import wordnet

from nltk.corpus import stopwords
import string

from proprocessing_final import process_tweets, process_tweet


params = dict(remove_USER_URL=False,
              remove_stopwords=True,
              remove_punctuation=True,
              appostrophe_handling=True,
              lemmatize=True,
              reduce_lengthenings=True,
              segment_words=True,
              correct_spelling=True,
              remove_hashtags=True,
              sym_spell=None,
        )


if __name__ == "__main__":

    testdataA = pd.read_csv("testset-levela.tsv", delimiter='\t')
    testdataB = pd.read_csv("testset-levelb.tsv", delimiter='\t')
    testdataC = pd.read_csv("testset-levelc.tsv", delimiter='\t')

    labelsA = pd.read_csv("labels-levela.csv", header=None, names=['id', 'label'])
    labelsB = pd.read_csv("labels-levelb.csv", header=None, names=['id', 'label'])
    labelsC = pd.read_csv("labels-levelc.csv", header=None, names=['id', 'label'])

    testdataA['tweet'] = process_tweets(testdataA['tweet'])
    testdataA['tweet'] = testdataA['tweet'].apply(lambda x: process_tweet(x, **params, trial=False))
    labelsA['label'] = labelsA['label'].apply(lambda x: 0 if x == 'NOT' else 1)

    testsetA = pd.merge(testdataA, labelsA, how="inner", on=["id", "id"])
    testsetA.to_csv('test_taskA.csv', encoding='utf-8')

    testdataB['tweet'] = process_tweets(testdataB['tweet'])
    testdataB['tweet'] = testdataA['tweet'].apply(lambda x: process_tweet(x, **params, trial=False))
    labelsB['label'] = labelsB['label'].apply(lambda x: 0 if x == 'UNT' else 1)

    testsetB = pd.merge(left=testdataB, right=labelsB, left_on='id', right_on='id')
    testsetB.to_csv('test_taskB.csv', encoding='utf-8')

    testdataC['tweet'] = process_tweets(testdataC['tweet'])
    testdataC['tweet'] = testdataC['tweet'].apply(lambda x: process_tweet(x, **params, trial=False))
    labelsC['label'] = labelsC.label.map({'IND': 0, 'OTH': 1, 'GRP': 2})

    testsetC = pd.merge(left=testdataC, right=labelsC, left_on='id', right_on='id')
    testsetC.to_csv('test_taskC.csv', encoding='utf-8')








