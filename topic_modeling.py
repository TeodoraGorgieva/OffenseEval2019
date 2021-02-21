import sys
import re, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer

import gensim
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

data = pd.read_csv("data/trainC.csv")
#print(data.head())


def strip_newline(series):
    return [review.replace('\n','') for review in series]

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def get_corpus(df):
    data['tweet'] = strip_newline(df.tweet)
    words = list(sent_to_words(df.tweet))
    words = remove_stopwords(words)
    bigram_mod = bigrams(words)
    bigram = [bigram_mod[review] for review in words]
    id2word = gensim.corpora.Dictionary(bigram)
    id2word.filter_extremes(no_below=10, no_above=0.35)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    return corpus, id2word, bigram



if __name__ == '__main__':

    train_corpus, train_id2word, bigram_train = get_corpus(data)

    lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus,
                                                id2word=train_id2word,
                                                num_topics=20,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    # print(lda_model.print_topics())

    train_vecs = []
    for i in range(len(data)):
        top_topics = lda_model.get_document_topics(train_corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(20)]
        topic_vec.extend([len(data.iloc[i].tweet)])  # length review
        train_vecs.append(topic_vec)

    X = np.array(train_vecs)
    y = np.array(data.subtask_c)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = SVC()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print(f"SVM accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"SVM precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"SVM recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"SVM F1 score: ", f1_score(y_test, y_predicted, average='macro'))

    # matrix = plot_confusion_matrix(clf, X_test, y_test)
    # plt.title('Confusion matrix for our classifier')
    # plt.show()

    # results = confusion_matrix(X_test, y_test)
    # print(results)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_predicted = gnb.predict(X_test)
    print(f"Gaussian Naive Bayes accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"Gaussian Naive Bayes precision: {precision_score(y_test, y_predicted, average='macro')}")
    print(f"Gaussian Naive Bayes recall score: {recall_score(y_test, y_predicted, average='macro')}")
    print(f"Gaussian Naive Bayes F1 score: {f1_score(y_test, y_predicted, average='macro')}")

    clf_knn = KNeighborsClassifier(n_neighbors=31)
    clf_knn.fit(X_train, y_train)
    y_predicted = clf_knn.predict(X_test)
    print(f"KNN accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"KNN  precision: {precision_score(y_test, y_predicted, average='macro')}")
    print(f"KNN  recall score: {recall_score(y_test, y_predicted, average='macro')}")
    print(f"KNN  F1 score: {f1_score(y_test, y_predicted, average='macro')}")

    lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
    lr.fit(X_train, y_train)
    y_predicted = lr.predict(X_test)
    print(f"Linear Regression accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"Linear Regression  precision: {precision_score(y_test, y_predicted, average='macro')}")
    print(f"Linear Regression  recall score: {recall_score(y_test, y_predicted, average='macro')}")
    print(f"Linear Regression  F1 score: {f1_score(y_test, y_predicted, average='macro')}")

    rfc = RandomForestClassifier(n_estimators=100,
                                 bootstrap=True,
                                 max_features='sqrt')
    rfc.fit(X_train, y_train)
    y_predicted = rfc.predict(X_test)
    print(f"Random Forest accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"Random Forest  precision: {precision_score(y_test, y_predicted, average='macro')}")
    print(f"Random Forest   recall score: {recall_score(y_test, y_predicted, average='macro')}")
    print(f"Random Forest   F1 score: {f1_score(y_test, y_predicted, average='macro')}")

