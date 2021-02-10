import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def word2vec_matrix(sentences, window):
    vocabulary = set()
    tokenized_sentences = []
    for sentence in sentences:
        words = sentence.split()
        tokenized_sentences.append(words)
        for word in words:
            vocabulary.add(word)

    model = Word2Vec(sentences=tokenized_sentences, size=50, window=window, min_count=1, sg=1)

    # save every word and every word's vector
    word_vector = list()
    index_name = list()
    for word in vocabulary:
        if word in model.wv.vocab.keys():
            word_vector.append(list(model.wv[word]))
            index_name.append(word)

    matrix = pd.DataFrame(word_vector, columns=[i for i in range(0, len(word_vector[0]))], index=index_name)
    return matrix


def create_doc_vector(corpus, matrix):
    new_corpus = []
    for item in corpus:
        tmp_df = pd.DataFrame()
        for word in item.split(' '):
            if word in matrix.index:
                word_vec = matrix.loc[word]
                tmp_df = tmp_df.append(pd.Series(word_vec), ignore_index=True)
        doc_vector = tmp_df.mean()
        new_corpus.append(list(doc_vector))
    return new_corpus


def create_train_test(task):
    train = pd.read_csv(f'train_task{task}_noemoji.csv', ',')
    train = train.copy().sample(1000, random_state=42)
    test = pd.read_csv(f'test_task{task}.csv')

    X_train = train['tweet'].values.astype('U')
    y_train = train[f'subtask_{task.lower()}']
    X_test = test['tweet'].values.astype('U')
    y_test = test['label']

    train_matrix = word2vec_matrix(X_train, 2)
    train_vectors = create_doc_vector(X_train, train_matrix)
    test_matrix = word2vec_matrix(X_test, 2)
    test_vectors = create_doc_vector(X_test, test_matrix)

    return train_vectors, y_train, test_vectors, y_test


def classify(train_vectors, y_train, test_vectors, y_test, task):
    print(f"Classification results for subtask {task}")

    clf = SVC()
    clf.fit(train_vectors, y_train)
    y_predicted = clf.predict(test_vectors)
    print(f"SVM accuracy: ", accuracy_score(y_test, y_predicted))
    print(f"SVM precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"SVM recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"SVM F1 score: ", f1_score(y_test, y_predicted, average='macro'))

    gnb = GaussianNB()
    gnb.fit(train_vectors, y_train)
    y_predicted = gnb.predict(test_vectors)
    print(f"Gaussian Bayes accuracy: ", accuracy_score(y_test, y_predicted))
    print(f"Gaussian Bayes precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"Gaussian Bayes recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"Gaussian Bayes F1 score: ", f1_score(y_test, y_predicted, average='macro'))

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

    train_vectors, y_train, test_vectors, y_test = create_train_test('A')
    classify(train_vectors, y_train, test_vectors, y_test, 'A')
    train_vectors, y_train, test_vectors, y_test = create_train_test('B')
    classify(train_vectors, y_train, test_vectors, y_test, 'B')
    train_vectors, y_train, test_vectors, y_test = create_train_test('C')
    classify(train_vectors, y_train, test_vectors, y_test, 'C')

