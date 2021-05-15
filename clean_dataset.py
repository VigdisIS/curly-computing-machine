import time

import nltk
import pandas as pd
import re
import preprocessor as p
from nltk.classify import svm
from nltk.corpus import stopwords

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

