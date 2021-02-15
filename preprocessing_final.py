import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from emoji.core import demojize
import num2words
import emoji

from symspellpy.symspellpy import SymSpell, Verbosity
import pandas as pd


def process_tweets(tweets_column):
    #tweets_column = tweets_column.apply(lambda x: emoji.demojize(x, delimiters=(" ", " ")))

    return tweets_column

def process_tweet(tweet,
                  remove_USER_URL=True,
                  remove_punctuation=True,
                  remove_stopwords=True,
                  remove_hashtags=True,
                  appostrophe_handling=True,
                  lemmatize=True,
                  sym_spell=SymSpell,
                  reduce_lengthenings=True,
                  segment_words=True,
                  correct_spelling=True,
                  ):
    """
    This function receives tweets and returns clean word-list
    """
    ### REMOVE HASHTAGS? #####################################################
    if remove_hashtags:
        tweet = re.sub(r'#\w+ ?', '', tweet)

    ### Handle USERS and URLS ################################################
    if remove_USER_URL:
       tweet = re.sub(r"@USER", "<user>", tweet)
       tweet = re.sub(r"URL", "<url>", tweet)


    ### Convert to lower case: Hi->hi, MAGA -> maga ##########################
    tweet = tweet.lower()

    ### Cleaning: non-ASCII filtering, some appostrophes, separation #########
    tweet = re.sub(r"’", r"'", tweet)
    #tweet = re.sub(r"[^A-Za-z0-9'^,!.\/+-=@]", " ", tweet)
    tweet = re.sub(r",", " ", tweet)
    tweet = re.sub(r"\.", " ", tweet)
    tweet = re.sub(r"!", " ", tweet)
    tweet = re.sub(r"\/", " ", tweet)
    tweet = re.sub(r"\^", " ", tweet)
    tweet = re.sub(r"\+", " ", tweet)
    tweet = re.sub(r"\*", "", tweet)
    tweet = re.sub(r"\-", " ", tweet)
    tweet = re.sub(r"\=", " ", tweet)
    tweet = re.sub(r":", " ", tweet)
    tweet = re.sub(r";", " ", tweet)
    tweet = re.sub(r"\"", " ", tweet)
    tweet = re.sub(r"“", " ", tweet)
    tweet = re.sub(r"\&", " ", tweet)
    tweet = re.sub(r"\?", " ", tweet)
    tweet = re.sub(r"\'", " ", tweet)
    tweet = re.sub(r"\(", " ", tweet)
    tweet = re.sub(r"\)", " ", tweet)
    tweet = re.sub(r"-", " ", tweet)
    tweet = re.sub(r"”", " ", tweet)
    tweet = re.sub(r",", " ", tweet)
    tweet = re.sub(r"%", " ", tweet)
    tweet = re.sub(r"‘", " ", tweet)
    tweet = re.sub(r"@", " ", tweet)
    tweet = re.sub(r"\$", " ", tweet)
    tweet = re.sub(r"–", " ", tweet)
    tweet = re.sub(r"£", " ", tweet)
    tweet = re.sub(r"{", " ", tweet)
    tweet = re.sub(r"}", " ", tweet)
    tweet = re.sub(r"・", " ", tweet)
    tweet = re.sub(r"\.{3}", " ", tweet)
    tweet = re.sub(r"„", " ", tweet)
    tweet = re.sub(r"\"", " ", tweet)
    tweet = re.sub(r"\s{2,}", " ", tweet)

    tweet = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), tweet)

    #tweet = emoji.demojize(tweet, delimiters=(" ", " "))

    tweet= re.sub(emoji.get_emoji_regexp(), r"", tweet)

    ### Remove Punctuation ###################################################
    #if remove_punctuation:
        #translator = str.maketrans('', '', ''.join(list(set(string.punctuation) - set("'"))))
        #tweet = tweet.translate(translator)

    # Tokenize sentence for further word-level processing
    tokenizer = TweetTokenizer()
    words = tokenizer.tokenize(tweet)

    ### Apostrophe handling:    you're   -> you are  ########################
    APPO = {"aren't": "are not", "can't": "cannot", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
            "i'd": "I would", "i'll": "I will", "i'm": "I am", "isn't": "is not", "it's": "it is", "it'll": "it will",
            "i've": "I have", "let's": "let us", "mightn't": "might not", "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
            "she'll": "she will", "she's": "she is", "shouldn't": "should not", "that's": "that is", "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are", "they've": "they have", "we'd": "we would", "we're": "we are", "weren't": "were not",
            "we've": "we have", "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have", "where's": "where is",
            "who'd": "who would", "who'll": "who will", "who're": "who are", "who's": "who is", "who've": "who have", "won't": "will not",
            "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have", "'re": " are",
            "wasn't": "was not", "we'll": " will"}
    if appostrophe_handling:
        words = [APPO[word] if word in APPO else word for word in words]

    tweet = ' '.join(words)
    words = tokenizer.tokenize(tweet)

    ### Lemmatisation:          drinking -> drink ###########################
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word, "v") for word in words]

    ### Remove stop words:      is, that, the, ... ##########################
    if remove_stopwords:
        eng_stopwords = set(stopwords.words("english"))
        words = [w for w in words if not w in eng_stopwords]

    ### Reduce lengthening:    aaaaaaaaah -> aah, bleeeeerh -> bleerh #################
    if reduce_lengthenings:
        pattern = re.compile(r"(.)\1{2,}")
        words = [pattern.sub(r"\1\1", w) for w in words]



    clean_tweet = " ".join(words)
    clean_tweet = re.sub("  ", " ", clean_tweet)
    clean_tweet = clean_tweet.lower()

    return clean_tweet


params = dict(remove_USER_URL=True,
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


def setA():

    data = pd.read_csv("olid-training-v1.0.tsv", sep='\t')
    data = data[['tweet', 'subtask_a']]
    data['subtask_a'] = data['subtask_a'].apply(lambda x: 0 if x == 'NOT' else 1)

    data['tweet'] = data['tweet'].apply(lambda x: process_tweet(x, **params))

    data.to_csv('data/trainA.csv', encoding='utf-8')


def setB():

    data = pd.read_csv("olid-training-v1.0.tsv", sep='\t')
    data = data[['tweet', 'subtask_b']]

    data['subtask_b'] = data['subtask_b'].fillna("NULL")
    mapper = {'UNT': 0, 'TIN': 1, 'NULL': 'NULL'}
    data['subtask_b'] = data.subtask_b.map(mapper)
    data = data.loc[data['subtask_b'] != 'NULL']

    data['tweet'] = data['tweet'].apply(lambda x: process_tweet(x, **params))
    data.to_csv('data/trainB.csv', encoding='utf-8')


def setC():
    data = pd.read_csv("olid-training-v1.0.tsv", sep='\t')
    data = data[['tweet', 'subtask_c']]

    data['subtask_c'] = data['subtask_c'].fillna("NULL")
    mapper = {'IND': 0, 'OTH': 1, 'GRP': 2, 'NULL': 'NULL'}
    data['subtask_c'] = data.subtask_c.map(mapper)
    data = data.loc[data['subtask_c'] != 'NULL']

    data['tweet'] = data['tweet'].apply(lambda x: process_tweet(x, **params))
    data.to_csv('train_taskC_noemoji.csv', encoding='utf-8')



if __name__ == "__main__":

    #setA()
    setB()
    setC()


