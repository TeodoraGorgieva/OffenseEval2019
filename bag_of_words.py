import nltk
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def frequency_dict(corpus, freq):
    for sentence in corpus:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in freq.keys():
                freq[token] = 1
            else:
                freq[token] += 1
    return freq


def bag_of_words(data_train, data_test):
    freq = frequency_dict(data_train, {})
    freq = frequency_dict(data_test, freq)
    most_freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:2000])

    sentence_vectors_train = []
    for sentence in data_train:
        sentence_tokens = nltk.word_tokenize(sentence)
        sent_vec = []
        for token in most_freq:
            if token in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors_train.append(sent_vec)

    sentence_vectors_test = []
    for sentence in data_test:
        sentence_tokens = nltk.word_tokenize(sentence)
        sent_vec = []
        for token in most_freq:
            if token in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors_test.append(sent_vec)
    return sentence_vectors_train, sentence_vectors_test


def create_train_test(task):
    train = pd.read_csv(f'train_task{task}_noemoji.csv', ',')
    train = train.copy().sample(1000, random_state=42)
    test = pd.read_csv(f'test_task{task}.csv')
    test = test.copy()

    X_train = train['tweet'].values.astype('U')
    y_train = train[f'subtask_{task.lower()}']
    X_test = test['tweet'].values.astype('U')
    y_test = test['label']

    X_train, X_test = bag_of_words(X_train, X_test)

    return X_train, y_train, X_test, y_test


def classify(X_train, y_train, X_test, y_test, task):
    print(f"Classification results for subtask {task}")

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_predicted = gnb.predict(X_test)
    print(f"Gaussian Naive Bayes accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"Gaussian Naive Bayes precision: {precision_score(y_test, y_predicted, average=None)}")
    print(f"Gaussian Naive Bayes recall score: {recall_score(y_test, y_predicted, average=None)}")
    print(f"Gaussian Naive Bayes F1 score: {f1_score(y_test, y_predicted, average=None)}")

    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print(f"Support Vector Classification accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"Support Vector Classification precision: {precision_score(y_test, y_predicted, average=None)}")
    print(f"Support Vector Classification recall score: {recall_score(y_test, y_predicted, average=None)}")
    print(f"Support Vector Classification F1 score: {f1_score(y_test, y_predicted, average=None)}")

    clf_knn = KNeighborsClassifier(n_neighbors=31)
    clf_knn.fit(X_train, y_train)
    y_predicted = clf_knn.predict(X_test)
    print(f"KNN accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"KNN  precision: {precision_score(y_test, y_predicted, average=None)}")
    print(f"KNN  recall score: {recall_score(y_test, y_predicted, average=None)}")
    print(f"KNN  F1 score: {f1_score(y_test, y_predicted, average=None)}")

    lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
    lr.fit(X_train, y_train)
    y_predicted = lr.predict(X_test)
    print(f"Linear Regression accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"Linear Regression  precision: {precision_score(y_test, y_predicted, average=None)}")
    print(f"Linear Regression  recall score: {recall_score(y_test, y_predicted, average=None)}")
    print(f"Linear Regression  F1 score: {f1_score(y_test, y_predicted, average=None)}")


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = create_train_test('A')
    classify(X_train, y_train, X_test, y_test, 'A')
    X_train, y_train, X_test, y_test = create_train_test('B')
    classify(X_train, y_train, X_test, y_test, 'B')
    X_train, y_train, X_test, y_test = create_train_test('C')
    classify(X_train, y_train, X_test, y_test, 'C')
