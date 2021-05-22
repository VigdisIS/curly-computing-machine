# Source used: https://www.pluralsight.com/guides/building-a-twitter-sentiment-analysis-in-python


# IMPORTS


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Stop words from NLTK
stop_words = set(stopwords.words('english'))


# METHODS


def preprocess_tweet_text(tweet):
    """ Normalizes the text in a tweet """

    # Lowers text
    tweet = tweet.lower()

    # Removes urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)

    # Removes user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', tweet)

    # Removes punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))

    # Removes extra 'twitter-specific' words not caught from other normalization
    stop_words.add('rt')
    stop_words.add('im')
    stop_words.add('u')

    # Removes stopwords, tokenizes
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]

    # Lemmatizes tokenized text
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]

    # Returns the joined result from the preprocessing
    return " ".join(lemma_words)


def load_dataset(filename, cols):
    """ Loads a dataset from a csv file """

    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset


def get_feature_vector(train_fit):
    """ Feature extraction using TFIDFVectorizer """

    # Loads TFIDFVectorizer with sublinear_tf, normalising bias against lengthy documents vs short documents
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector


# CODE


# Loading Sentiment140 dataset from 'datasets' folder
dataset = load_dataset('../datasets/sentiment_dataset.csv', ['target', 'user', 'text', 'cleaned_text'])

# Bar chart showing the balanced dataset
dataset.groupby('target').count().plot.bar(ylim=0)
plt.show()

# Preprocessing the tweets in the dataset
dataset.text = dataset['text'].apply(preprocess_tweet_text)

# Extracting features from dataset using the 'get_feature_vector' method
tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
x = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
y = np.array(dataset.iloc[:, 0]).ravel()

# Splitting dataset into train, test with 80:20 ratio
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# Training the Logistics Regression model classifier
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(x_train, y_train)
y_predict_lr = LR_model.predict(x_test)

# Accuracy of model
print(accuracy_score(y_test, y_predict_lr))

# Classification report
print(classification_report(y_test, LR_model.predict(x_test)))

# pickling the vectorizer
pickle.dump(tf_vector, open('sent_vectorizer.sav', 'wb'))
# pickling the model
pickle.dump(LR_model, open('sent_classifier.sav', 'wb'))
