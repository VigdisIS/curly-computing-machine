# IMPORTS


import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from nltk import PorterStemmer, re
from nltk.corpus import stopwords
from models.sentiment_classifier import load_dataset

# Stemmer from NLTK
stemmer = PorterStemmer()

# Stop words from NLTK
stop_words = set(stopwords.words('english'))

# Loads the pickled vectorizer and classifier model
tf_vector = pickle.load(open('../models/sent_vectorizer.sav', 'rb'))
LR_model = pickle.load(open('../models/sent_classifier.sav', 'rb'))


# METHOD


def plot_sphere_sentiment(path):
    """ Method to iterate through each sphere and accounts within each sphere to predict labels and create grouped
            bar charts for each sphere and overall """

    # Keeps track of all counts of positive and negative tweets from each sphere
    pos_tweets = []
    neg_tweets = []

    # Array to plot overall sentiment grouped bar chart, which will contain the array of count of positive tweets and
    # the array of count of negative tweets from each sphere
    data = []

    for tweetset in glob.glob(path):
        """ Iterates through each sphere, keeping track of the amount of positive and negative tweets to create a 
                sphere-specific grouped bar chart for current sphere """

        # Count of positive and negative tweets in current sphere, for use in the overall sentiment grouped bar chart
        pos_count = 0
        neg_count = 0

        # Sphere-specific grouped bar chart values
        sphere_fig = plt.figure()
        sphere_x = np.arange(5)
        sphere_ax = sphere_fig.add_subplot(111)

        # Array of accounts within sphere, for use when labeling chart
        accounts = []

        # Keeps track of all counts of positive and negative tweets in current sphere
        sphere_pos_tweets = []
        sphere_neg_tweets = []

        # Array to plot sphere-specific bar chart, which will contain the array of count of positive tweets and the
        # array of count of negative tweets
        sphere_data = []

        # All accounts (datasets) in current sphere to iterate over
        path = tweetset + "/*.csv"

        for fname in glob.glob(path):
            """ Iterates through each account (and respective dataset with tweets), preprocesses, extracts features 
                    and predicts labels of the accounts' tweets """

            # Loads tweet sphere dataset using 'load_dataset' method from 'sentiment_classifier.py'
            dataset = load_dataset(fname, ['username', 'text', 'cleaned_text'])

            # Preprocesses text
            dataset.text = dataset['text'].apply(lambda x: " ".join(
                [stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stop_words]).lower())
            tweet_text = np.array(dataset.iloc[:, 1]).ravel()

            # Extracts features from tweet text
            tweet_text_feature = tf_vector.transform(np.array(tweet_text).ravel())

            # Using Logistic Regression model to predict labels
            labeled_tweet_text = LR_model.predict(tweet_text_feature)

            # Counts occurrences of positive and negative labeled tweets
            pos_occurrences = np.count_nonzero(labeled_tweet_text == 4)
            neg_occurrences = np.count_nonzero(labeled_tweet_text == 0)

            # Appends the counts of positive and negative tweets to respective arrays
            sphere_pos_tweets.append(pos_occurrences)
            sphere_neg_tweets.append(neg_occurrences)

            # Appends the current account to the sphere accounts array to use in the sphere-specific bar chart
            accounts.append(dataset.iloc[:, 0][1])

            # Updates count of total pos and neg tweets for all accounts in sphere
            pos_count += pos_occurrences
            neg_count += neg_occurrences

        # Appends count of total pos and neg tweets of sphere for use in sphere-specific bar chart
        sphere_data.append(sphere_pos_tweets)
        sphere_data.append(sphere_neg_tweets)

        # Plots the sphere-specific grouped bar chart with the counts of pos and neg tweets in the 'sphere_data' array
        sphere_ax.bar(sphere_x + 0.00, sphere_data[0], color='mediumseagreen',
                      width=0.5)  # Positive tweets from each account
        sphere_ax.bar(sphere_x + 0.3, sphere_data[1], color='indianred', width=0.5)  # Negative tweets from each account

        # Sets the title of the chart to the current Twitter sphere (from path)
        sphere_ax.set_title(tweetset.split('\\')[1])

        # Sets chart labels
        sphere_ax.set_ylabel('Tweets')
        sphere_ax.set_xlabel('Accounts')

        # Sets ticks to amount of tweets and names of the different accounts
        sphere_ax.set_xticks(sphere_x + 0.15)
        sphere_ax.set_xticklabels(tuple(accounts))

        # Sets the labels positive and negative for the different bars
        sphere_ax.legend(labels=['positive', 'negative'])

        # Creates the sphere-specific grouped bar chart
        plt.show()

        # Appends total count of all positive and negative tweets in current sphere to respective arrays for overall
        # sentiment bar chart
        pos_tweets.append(pos_count)
        neg_tweets.append(neg_count)

    # Appends count of all positive and negative tweets from each sphere to plot overall sentiment bar chart
    data.append(pos_tweets)
    data.append(neg_tweets)

    # Overall sentiment grouped bar chart values
    fig = plt.figure()
    x = np.arange(5)
    ax = fig.add_subplot(111)

    # Plots the overall sentiment grouped bar chart with the counts of pos and neg tweets from the 'data' array
    ax.bar(x + 0.00, data[0], color='mediumseagreen', width=0.5)  # Positive tweets from each sphere
    ax.bar(x + 0.3, data[1], color='indianred', width=0.5)  # Negative tweets from each sphere

    # Sets title, labels and ticks for chart
    ax.set_title('Sentiment')
    ax.set_ylabel('Tweets')
    ax.set_xlabel('Communities')
    ax.set_xticks(x + 0.15)
    ax.set_xticklabels(('eSports', 'NSFW', 'politicians', 'scientists', 'streamers'))
    ax.legend(labels=['positive', 'negative'])

    # Creates the overall sentiment grouped bar chart
    plt.show()


# Calling the method to predict labels and generate grouped bar charts with result
plot_sphere_sentiment('../tweets/*')
