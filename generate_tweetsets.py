import unicodecsv as unicodecsv
#from text_unidecode import unidecode
from tweepy import OAuthHandler, API, Cursor

import time

import os

CONS_KEY = os.getenv('CONS_KEY')
CONS_SECRET = os.getenv('CONS_SECRET')
ACC_TOKEN = os.getenv('ACC_TOKEN')
ACC_TOKEN_SECRET = os.getenv('ACC_TOKEN_SECRET')

# Personal tokens
from unidecode import unidecode

consumer_key = CONS_KEY
consumer_secret = CONS_SECRET
access_token = ACC_TOKEN
access_token_secret = ACC_TOKEN_SECRET

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = API(auth)

# Usernames whose tweets we want to gather.
users = []

for user in users:

    user_obj = api.get_user(user)

    # Gather info specific to the current user.
    user_info = [user_obj.screen_name]

    with open(f'{user_info[0]}.csv', 'wb') as file:
        writer = unicodecsv.writer(file, delimiter=',', quotechar='"')

        # Write header row.
        header = ['username', 'text']

        writer.writerow(header)

        # Get 1000 most recent tweets for the current user.
        for tweet in Cursor(api.user_timeline, screen_name=user).items(1000):
            # Get info specific to the current tweet of the current user.
            tweet_text = [unidecode(tweet.text)]

            writer.writerow(user_info + tweet_text)


    # Show progress.
    print("Wrote tweets by %s to CSV." % user)
    time.sleep(10)