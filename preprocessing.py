import pandas as pd
import matplotlib.pyplot as plt
import re
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


df = pd.read_csv("twitter_data.csv" , encoding='latin-1')
df.columns=['Sentiment', 'id', 'Date', 'Query', 'User', 'Tweet']
df = df.drop(columns=['id', 'Date', 'Query', 'User'], axis=1)

#0: negative sentiment, 1:positive sentiment
df['Sentiment'] = df['Sentiment'].apply(lambda x: 0 if x == 0 else 1)

df = df.copy().sample(8000, random_state=42)

#contractins in english language
contractions = pd.read_csv('contractions.csv', index_col='Contraction')
contractions.index = contractions.index.str.lower()
contractions.Meaning = contractions.Meaning.str.lower()
contractions_dict = contractions.to_dict()['Meaning']

def preprocess_apply(tweet):

    """
        Function for text cleaning
        :param text: raw text
        :return: preprocessed text
    """

    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    hashtagPattern = '#[^\s]+'
    alphaPattern = "[^a-z0-9<>]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    tweet = tweet.lower()
    # transform url adress to "url" token
    tweet = re.sub(urlPattern, 'url', tweet)
    # transform each @username to "at_user" token
    tweet = re.sub(userPattern, 'atuser', tweet)

    # transform hastags ex: #love to "love" token
    tweet = tweet.replace("#", "")

    # transform contractions to their full meaning
    for contraction, replacement in contractions_dict.items():
        tweet = tweet.replace(contraction, replacement)

    # Remove non-alphanumeric and symbols
    tweet = re.sub(alphaPattern, ' ', tweet)

    # Adding space on either side of '/' to seperate words
    tweet = re.sub(r'/', ' / ', tweet)

    return tweet


def stemSentence(text):
    """
    Function for stemming text
    :param text: raw text
    :return: stemmed sentences
    """
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')

    stem_sentence = []
    for word in text.split():
        if word not in stop_words and word not in string.punctuation:
            if len(word) > 2:
                stem_sentence.append(stemmer.stem(word))

    return " ".join(stem_sentence)


def lemmatize_sentence(text):
    """
    Function to lemmatize text.
    If a tag starts with NN, the word is a noun and if it stars with VB, the word is a verb
    :param text: raw text
    :return: lemmatized sentences
    """
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    lemmatized_sentence = []
    for word, tag in pos_tag(text.split()):

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        if word not in stop_words and word not in string.punctuation:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))

    return " ".join(lemmatized_sentence)


df['processed_tweet'] = df.Tweet.apply(preprocess_apply)
df.processed_tweet = df.processed_tweet.apply(lemmatize_sentence)

df['Words'] = df['processed_tweet'].apply(lambda x:str(x).split())

#most frequent positive words
top_pos = Counter([word for text in df[df['Sentiment']==1]['Words'] for word in text])
top_pos_df=pd.DataFrame(top_pos.most_common(15),columns=['Words','Counts'])
#print(top_pos_df)

#most frequent negative words
top_neg = Counter([word for text in df[df['Sentiment']==0]['Words'] for word in text])
top_neg_df=pd.DataFrame(top_pos.most_common(15),columns=['Words','Counts'])
#print(top_neg_df)

def word_cloud(array_text, color='white'):
    """
    Function to generate word_cloud
    :param array_text:
    :param color:
    :return:
    """

    wc = WordCloud(max_words=100,
                   background_color=color,
                   width=1600, height=800,
                   collocations=False).generate(" ".join(array_text))
    plt.figure(figsize=(20, 20))
    plt.imshow(wc)
    plt.show()


#word clouds

#wc_pos = word_cloud(top_pos)
#wc_neg = word_cloud(top_neg, 'black')



