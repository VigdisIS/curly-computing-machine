import time

import nltk
import pandas as pd
import re
import preprocessor as p
from nltk.classify import svm
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

'''
tweets_pre = pd.read_csv('./datasets/sentiment140.csv')

tweets_pre.to_csv('./datasets/sentiment140.csv', header=['target', 'id', 'date', 'flag', 'user', 'text'], index=False)

keep_col = ['target','user','text']
tweets = tweets_pre[keep_col]
tweets.to_csv('./datasets/sentiment140.csv', index=False)

'''


tweets = pd.read_csv('./datasets/sentiment140.csv')



from sklearn.utils import shuffle

tweets = shuffle(tweets)


tweets.drop(tweets.tail(1580000).index, inplace=True)

tweets.to_csv('./datasets/sentiment_dataset.csv', index=False)

''''''
'''
x = tweets['text']



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

cleaned_tweets = clean_tweets(x)

tweets['cleaned_text'] = cleaned_tweets

tweets.to_csv('./datasets/sentiment_dataset.csv', index=False)

'''