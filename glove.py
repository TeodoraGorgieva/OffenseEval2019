import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def create_glove_vocabulary(file_name):
    vocabulary = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            word = line.split()[0]
            vector = np.asarray(line.split()[1:], dtype='float32')
            vocabulary.setdefault(word, vector)
    return vocabulary


def glove_representation(corpus, vocabulary, labels):
    new_corpus, new_labels = [], []
    for i, item in enumerate(corpus):
        tmp_df = pd.DataFrame()
        for word in item.split():
            if word in vocabulary:
                word_vec = vocabulary[word]
                tmp_df = tmp_df.append(pd.Series(word_vec), ignore_index=True)
        doc_vector = tmp_df.mean()
        if len(doc_vector) == 0:
            continue
        new_corpus.append(list(doc_vector))
        new_labels.append(labels.iloc[i])
    return new_corpus, new_labels


if __name__ == '__main__':
    train = pd.read_csv('train_taskA.csv', ',')
    train = train.copy().sample(1000, random_state=42)
    test = pd.read_csv('test_taskA.csv')
    test = test.copy()

    glove_dict = create_glove_vocabulary('glove.6B.50d.txt')

    y_train = train['subtask_a']
    y_test = pd.read_csv('labels-levela.csv', header=None, names=['id', 'label'])
    y_test = y_test['label'].apply(lambda x: 0 if x == 'NOT' else 1)

    train_vectors, y_train = glove_representation(train['tweet'].values.astype('U'), glove_dict, y_train)
    test_vectors, y_test = glove_representation(test['tweet'].values.astype('U'), glove_dict, y_test)

    clf = SVC()
    clf.fit(train_vectors, y_train)
    y_predicted = clf.predict(test_vectors)
    print(f"SVM accuracy: ", accuracy_score(y_test, y_predicted))
    print(f"SVM precision: ", precision_score(y_test, y_predicted, average=None))
    print(f"SVM recall score: ", recall_score(y_test, y_predicted, average=None))
    print(f"SVM F1 score: ", f1_score(y_test, y_predicted, average=None))

    gnb = GaussianNB()
    gnb.fit(train_vectors, y_train)
    y_predicted = gnb.predict(test_vectors)
    print(f"Gaussian Bayes accuracy: ", accuracy_score(y_test, y_predicted))
    print(f"Gaussian Bayes precision: ", precision_score(y_test, y_predicted, average=None))
    print(f"Gaussian Bayes recall score: ", recall_score(y_test, y_predicted, average=None))
    print(f"Gaussian Bayes F1 score: ", f1_score(y_test, y_predicted, average=None))

    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(train_vectors, y_train)
    y_predicted = clf_knn.predict(test_vectors)
    print(f"KNN accuracy: ", accuracy_score(y_test, y_predicted))
    print(f"KNN  precision: ", precision_score(y_test, y_predicted, average=None))
    print(f"KNN  recall score: ", recall_score(y_test, y_predicted, average=None))
    print(f"KNN  F1 score: ", f1_score(y_test, y_predicted, average=None))

    lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
    lr_tfidf = lr.fit(train_vectors, y_train)
    y_predicted = lr.predict(test_vectors)
    print(f"Linear Regression accuracy: ", accuracy_score(y_test, y_predicted))
    print(f"Linear Regression  precision: ", precision_score(y_test, y_predicted, average=None))
    print(f"Linear Regression  recall score: ", recall_score(y_test, y_predicted, average=None))
    print(f"Linear Regression  F1 score: ", f1_score(y_test, y_predicted, average=None))
