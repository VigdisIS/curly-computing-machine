import glob
from dbm.ndbm import library

import nltk
import pandas as pd
import unicodecsv
from numpy.distutils.command.install import install


from models.sentiment_classifier import load_dataset, preprocess_tweet_text


def most_common_words(path):
    print(path)

    spheres = []
    freq_words = []

    for tweetset in glob.glob(path):

        sphere = tweetset.split('\\')[1]

        corpus = ""

        path = tweetset + "/*.csv"

        for fname in glob.glob(path):
            print(fname)

            # Load dataset
            dataset = load_dataset(fname, ['username', 'text', 'cleaned_text'])
            # Preprocess data
            dataset.text = dataset['text'].apply(preprocess_tweet_text)

            corpus += ' '.join(dataset.iloc[:, 1])

        allWords = nltk.tokenize.word_tokenize(corpus)
        allWordDist = nltk.FreqDist(allWords)

        mostCommon = allWordDist.most_common(10)
        print(mostCommon)

        spheres.append(sphere)
        words = [w for w, _ in mostCommon]
        freq_words.append(words)

    with open('word_freq_corpus.csv', 'wb') as file:
        writer = unicodecsv.writer(file, delimiter=',', quotechar='"')

        # Write header row.
        header = ['sphere', 'text']

        writer.writerow(header)

        print(spheres[0])
        print(freq_words[0])

        # Get 1000 most recent tweets for the current user.
        for i in range(len(spheres)):
            sphere = [spheres[i]]
            words = [' '.join(freq_words[i])]

            writer.writerow(sphere + words)


most_common_words('../tweets/*')

