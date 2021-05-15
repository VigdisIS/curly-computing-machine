import glob
import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from models.sentiment_classifier import clean_tweets

# load the model from disk
tf_vector = pickle.load(open('../models/emot_vectorizer.sav', 'rb'))
LR_model = pickle.load(open('../models/emot_classifier.sav', 'rb'))

tweets = pd.read_csv('../tweets/politicians/mindyfinn.csv')

tweet_text = tweets['text']

preprocessed_text = clean_tweets(tweet_text)

test_feature = tf_vector.transform(np.array(preprocessed_text).ravel())

# Using Logistic Regression model for prediction
test_prediction_lr = LR_model.predict(test_feature)
test_prediction_prob = LR_model.predict_proba(test_feature)

print(test_prediction_lr)
print(test_prediction_prob)