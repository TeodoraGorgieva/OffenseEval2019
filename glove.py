import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
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


def glove_representation(corpus, labels, glove_dict):
    new_corpus, new_labels = [], []
    for i, item in enumerate(corpus):
        tmp_df = pd.DataFrame()
        for word in item.split():
            if word in glove_dict:
                word_vec = glove_dict[word]
            else:
                # unknown words are represented with zeros
                word_vec = [0] * 200
            tmp_df = tmp_df.append(pd.Series(word_vec), ignore_index=True)
        doc_vector = tmp_df.mean()
        new_corpus.append(list(doc_vector))
        new_labels.append(labels[i])
    return new_corpus, new_labels


def create_train_test(task, glove_dict):
    file = f'data/train{task}.csv'
    data = pd.read_csv(file, ',')

    text = data['tweet'].values.astype('U')
    labels = list(data[f'subtask_{task.lower()}'])
    test_vectors, labels = glove_representation(text, labels, glove_dict)
    X_train, X_test, y_train, y_test = train_test_split(test_vectors, labels, train_size=0.8, random_state=42)

    return X_train, X_test, y_train, y_test


def classify(train_vectors, y_train, test_vectors, y_test, task):
    print(f'Classification results for subtask {task}')

    clf = SVC()
    clf.fit(train_vectors, y_train)
    y_predicted = clf.predict(test_vectors)
    print(f"SVM accuracy: ", accuracy_score(y_test, y_predicted))
    print(f"SVM precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"SVM recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"SVM F1 score: ", f1_score(y_test, y_predicted, average='macro'))

    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(train_vectors, y_train)
    y_predicted = clf_knn.predict(test_vectors)
    print(f"KNN accuracy: ", accuracy_score(y_test, y_predicted))
    print(f"KNN  precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"KNN  recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"KNN  F1 score: ", f1_score(y_test, y_predicted, average='macro'))

    lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
    lr.fit(train_vectors, y_train)
    y_predicted = lr.predict(test_vectors)
    print(f"Linear Regression accuracy: ", accuracy_score(y_test, y_predicted))
    print(f"Linear Regression  precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"Linear Regression  recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"Linear Regression  F1 score: ", f1_score(y_test, y_predicted, average='macro'))


if __name__ == '__main__':
    glove_dict = create_glove_vocabulary('glove.twitter.27B.200d.txt')

    train_vectors, test_vectors, y_train, y_test = create_train_test('A', glove_dict)
    classify(train_vectors, y_train, test_vectors, y_test, 'A')
    train_vectors, test_vectors, y_train, y_test = create_train_test('B', glove_dict)
    classify(train_vectors, y_train, test_vectors, y_test, 'B')
    train_vectors, test_vectors, y_train, y_test = create_train_test('C', glove_dict)
    classify(train_vectors, y_train, test_vectors, y_test, 'C')
