# Source used: https://towardsdatascience.com/tweepy-for-beginners-24baf21f2c25


# IMPORTS


import unicodecsv as unicodecsv
import os
from tweepy import OAuthHandler, API, Cursor
from unidecode import unidecode


# CODE


''' Code to generate datasets consisting of tweets from users with Tweepy API '''

# Personal tokens from .env to access Twitter's API
CONS_KEY = os.getenv('CONS_KEY')
CONS_SECRET = os.getenv('CONS_SECRET')
ACC_TOKEN = os.getenv('ACC_TOKEN')
ACC_TOKEN_SECRET = os.getenv('ACC_TOKEN_SECRET')

consumer_key = CONS_KEY
consumer_secret = CONS_SECRET
access_token = ACC_TOKEN
access_token_secret = ACC_TOKEN_SECRET

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = API(auth)


# Usernames of five accounts from each sphere to gather tweets from
users = [
    'G2Caps', 'G2Jankos', 'GeT_RiGhT', 'jakeow', 'OG_BDN0tail',  # eSports
    'Brooklyn_Chase', 'lisadaniels3', 'MiaMalkova', 'rileyreidx3', 'thebonnierotten',  # NSFW
    'cbellatoni', 'EWErickson', 'JoeBiden', 'mindyfinn', 'nicopitney',  # politicians
    'BadAstronomer', 'bengoldacre', 'neiltyson', 'ProfBrianCox', 'RichardDawkins',  # scientists
    'Ninja', 'pokimanelol', 'TSM_Myth', 'Valkyrae', 'xQc'  # streamers
]

# Iterates through each user and gathering data through the API
for user in users:
    user_obj = api.get_user(user)

    # Extracts screen name from user
    user_info = [user_obj.screen_name]

    # Generates a dataset with 1000 tweets from the user
    with open(f'{user_info[0]}.csv', 'wb') as file:
        writer = unicodecsv.writer(file, delimiter=',', quotechar='"')

        # Writes the header row
        header = ['username', 'text']
        writer.writerow(header)

        # Iterates through the 1000 most recent tweets from the user
        for tweet in Cursor(api.user_timeline, screen_name=user).items(1000):
            # Gets the text of the tweet and writes to the file
            tweet_text = [unidecode(tweet.text)]
            writer.writerow(user_info + tweet_text)
