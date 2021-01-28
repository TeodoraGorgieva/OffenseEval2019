import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
import string
from nltk.corpus import wordnet as wn

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.stem import SnowballStemmer

from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


if __name__ == '__main__':

    tweets = pd.read_csv('tweets.csv', '\t', encoding='latin-1')
    train = tweets.copy().sample(10000, random_state=42)
    test = pd.read_csv('test_tweets.csv', encoding='latin-1')

    X_train = train['processed_tweet'].values.astype('U')
    X_test = test['processed_tweet'].values.astype('U')
    y_train = train['Sentiment']
    y_test = test['Sentiment']

    tv = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, use_idf=True, ngram_range=(1, 2))
    tv_X_train = tv.fit_transform(X_train)
    tv_X_test = tv.transform(X_test)

    print('TfidfVectorizer_train:', tv_X_train.shape)
    print('TfidfVectorizer_test:', tv_X_test.shape)

    tv_X_train = tv_X_train.toarray()
    tv_X_test = tv_X_test.toarray()

    clf = SVC()
    clf.fit(tv_X_train, y_train)
    y_predicted = clf.predict(tv_X_test)
    print(f"SVM accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"SVM precision: ", precision_score(y_test, y_predicted, average=None))
    print(f"SVM recall score: ", recall_score(y_test, y_predicted, average=None))
    print(f"SVM F1 score: ", f1_score(y_test, y_predicted, average=None))

    gnb = GaussianNB([0, 1])
    gnb.fit(tv_X_train, y_train)
    y_predicted = gnb.predict(tv_X_test)
    print(f"Gaussian Bayes accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"Gaussian Bayes precision: ", precision_score(y_test, y_predicted, average=None))
    print(f"Gaussian Bayes recall score: ", recall_score(y_test, y_predicted, average=None))
    print(f"Gaussian Bayes F1 score: ", f1_score(y_test, y_predicted, average=None))


    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(tv_X_train, y_train)
    y_predicted = clf_knn.predict(tv_X_test)
    print(f"KNN accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"KNN  precision: ", precision_score(y_test, y_predicted, average=None))
    print(f"KNN  recall score: ", recall_score(y_test, y_predicted, average=None))
    print(f"KNN  F1 score: ", f1_score(y_test, y_predicted, average=None))

    lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
    lr_tfidf = lr.fit(tv_X_train, y_train)
    y_predicted = lr.predict(tv_X_test)
    print(f"Linear Regression accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"Linear Regression  precision: ", precision_score(y_test, y_predicted, average=None))
    print(f"Linear Regression  recall score: ", recall_score(y_test, y_predicted, average=None))
    print(f"Linear Regression  F1 score: ", f1_score(y_test, y_predicted, average=None))
