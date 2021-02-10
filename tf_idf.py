import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


if __name__ == '__main__':
    train = pd.read_csv('train_taskA.csv', ',')
    train = train.copy().sample(1000, random_state=42)
    test = pd.read_csv('test_taskA.csv')

    X_train = train['tweet'].values.astype('U')
    X_test = test['tweet'].values.astype('U')
    y_train = train['subtask_a']
    y_test = test['label']

    tv = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, use_idf=True, ngram_range=(2, 7))
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
    print(f"SVM precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"SVM recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"SVM F1 score: ", f1_score(y_test, y_predicted, average='macro'))

    gnb = GaussianNB([0, 1])
    gnb.fit(tv_X_train, y_train)
    y_predicted = gnb.predict(tv_X_test)
    print(f"Gaussian Bayes accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"Gaussian Bayes precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"Gaussian Bayes recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"Gaussian Bayes F1 score: ", f1_score(y_test, y_predicted, average='macro'))

    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(tv_X_train, y_train)
    y_predicted = clf_knn.predict(tv_X_test)
    print(f"KNN accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"KNN  precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"KNN  recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"KNN  F1 score: ", f1_score(y_test, y_predicted, average='macro'))

    lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
    lr_tfidf = lr.fit(tv_X_train, y_train)
    y_predicted = lr.predict(tv_X_test)
    print(f"Linear Regression accuracy: {accuracy_score(y_test, y_predicted)}")
    print(f"Linear Regression  precision: ", precision_score(y_test, y_predicted, average='macro'))
    print(f"Linear Regression  recall score: ", recall_score(y_test, y_predicted, average='macro'))
    print(f"Linear Regression  F1 score: ", f1_score(y_test, y_predicted, average='macro'))
