import nltk
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


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


if __name__ == '__main__':
    train = pd.read_csv('tweets.csv', '\t')
    train = train.copy().sample(10000, random_state=42)
    test = pd.read_csv('test_tweets.csv')

    bag_of_words_train, bag_of_words_test = bag_of_words(train['processed_tweet'].values.astype('U'),
                                                         test['processed_tweet'].values.astype('U'))

    y_train = train['Sentiment']
    y_test = test['Sentiment']
    gnb = GaussianNB([0, 1])
    gnb.fit(bag_of_words_train, y_train)
    y_predicted = gnb.predict(bag_of_words_test)
    print(f"Naive Bayes accuracy: {accuracy_score(y_test, y_predicted)}")

    clf = svm.SVC()
    clf.fit(bag_of_words_train, y_train)
    y_predicted = clf.predict(bag_of_words_test)
    print(f"Support Vector Classification accuracy: {accuracy_score(y_test, y_predicted)}")
