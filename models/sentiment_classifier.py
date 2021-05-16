import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# ML Libraries
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Global Parameters
stop_words = set(stopwords.words('english'))
# tweets = pd.read_csv('../datasets/sentiment_dataset.csv')
# x = tweets['text']

# Source used: https://www.pluralsight.com/guides/building-a-twitter-sentiment-analysis-in-python

import matplotlib.pyplot as plt



def preprocess_tweet_text(tweet):
    tweet = tweet.lower()

    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))

    # Remove twitter words
    stop_words.add('rt')
    stop_words.add('im')
    stop_words.add('u')

    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]

    #ps = PorterStemmer()
    #stemmed_words = [ps.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]

    return " ".join(lemma_words)


def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset


def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset




# Load dataset
dataset = load_dataset('../datasets/sentiment_dataset.csv', ['target', 'user', 'text', 'cleaned_text'])


# Remove unwanted columns from dataset
dataset = remove_unwanted_cols(dataset, ['user', 'cleaned_text'])

dataset.groupby('target').count().plot.bar(ylim=0)
plt.show()

#Preprocess data
dataset.text = dataset['text'].apply(preprocess_tweet_text)


# Split dataset into Train, Test

# Same tf vector will be used for Testing sentiments on unseen trending data
tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
y = np.array(dataset.iloc[:, 0]).ravel()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Training Logistics Regression model
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(X_train, y_train)
y_predict_lr = LR_model.predict(X_test)

print(accuracy_score(y_test, y_predict_lr))
print(classification_report(y_test, LR_model.predict(X_test)))

import pickle
# pickling the vectorizer
pickle.dump(tf_vector, open('sent_vectorizer.sav', 'wb'))
# pickling the model
pickle.dump(LR_model, open('sent_classifier.sav', 'wb'))





