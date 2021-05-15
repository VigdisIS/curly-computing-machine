import string
import time

import nltk
import pandas as pd
import re
import preprocessor as p
from nltk import word_tokenize
from nltk.classify import svm
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

nltk.download('stopwords')
import re



'''
tweets_pre = pd.read_csv('./datasets/sentiment140.csv')

tweets_pre.to_csv('./datasets/sentiment140.csv', header=['target', 'id', 'date', 'flag', 'user', 'text'], index=False)

keep_col = ['target','user','text']
tweets = tweets_pre[keep_col]
tweets.to_csv('./datasets/sentiment140.csv', index=False)

'''


# set up punctuations we want to be replaced
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|("
                              "\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

def clean_tweets(df):
    tempArr = []
    for line in df:

       # send to tweet_processor
        tmpL = p.clean(line)

        # remove puctuation
        tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower())  # convert all tweets to lower cases
        tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
        tempArr.append(tmpL)
    return tempArr


def preprocess_tweet_text(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]

    # ps = PorterStemmer()
    # stemmed_words = [ps.stem(w) for w in filtered_words]
    # lemmatizer = WordNetLemmatizer()
    # lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

    return " ".join(filtered_words)


tweets = pd.read_csv('./tweets/eSports/OG_BDN0tail.csv')

cleaned = clean_tweets(tweets['text'])

tweets['cleaned_text'] = cleaned

tweets.to_csv('./tweets/eSports/OG_BDN0tail.csv', index=False)


'''
import glob

path = "./tweets/*"
for tweetset in glob.glob(path):
    print(tweetset)
    path = tweetset + "/*.csv"

    for fname in glob.glob(path):
        print(fname)

        tweets = pd.read_csv(fname)

        cleaned = clean_tweets(tweets['text'])

        tweets['cleaned_text'] = cleaned

        tweets.to_csv(fname, index=False)

'''