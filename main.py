import numpy as np
import pandas as pd
import re
import string
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import tweepy
import tokens
import time
import os
import sys
import json


files = os.listdir(".")

for filename in files:
    if re.search(r'vec\.pkl',filename):
        vectorizer_loc = filename
    elif re.search(r'mod\.pkl',filename):
        model_loc = filename
        

# loading classifier and fitted vectorizer 
mnb = joblib.load(model_loc)
tf = joblib.load(vectorizer_loc)

# connecting to twitter
auth = tweepy.OAuthHandler(tokens.consumer_key, tokens.consumer_secret)
auth.set_access_token(tokens.access_token, tokens.access_token_secret)

api = tweepy.API(auth,wait_on_rate_limit=True)

def teamSentiment(query):
    
    tweets = tweepy.Cursor(api.search,q=query,tweet_mode="extended").items(500)
    predictions = []

    # processing tweets and making predictions
    for tweet in tweets:
        try:
            #retweeted content
            tweet_txt = tweet.retweeted_status.full_text 
            removed_RT = re.sub(r'RT\s+',"",tweet_txt)
            removed_handle = re.sub(r'@\w+:?\s*',"",removed_RT)
            removed_link = re.sub(r'https?://.+\s*',"",removed_handle)
            line = filter(lambda x: x in string.printable, removed_link)
            
            vec = tf.transform([line])
            pred = mnb.predict(vec)
            predictions.append(int(pred))
       
        except (AttributeError, KeyError):
            #normal tweets
            if tweet.truncated == False and tweet.in_reply_to_status_id_str == None:
                tweet_txt = tweet.full_text 
                removed_RT = re.sub(r'RT\s+',"",tweet_txt)
                removed_handle = re.sub(r'@\w+:?\s*',"",removed_RT)
                removed_link = re.sub(r'https?://.+\s*',"",removed_handle)
                line = filter(lambda x: x in string.printable, removed_link)
                
                vec = tf.transform([line])
                pred = mnb.predict(vec)
                predictions.append(int(pred))
        
    # determining a tally of positive (1) and negative (0)
    predictions_np = np.array(predictions)
    prediction_tally = np.bincount(predictions_np)

    return prediction_tally[1]/float(prediction_tally[1]+prediction_tally[0])


teams = ["Atlanta Hawks","Boston Celtics","Brooklyn Nets","Charlotte Hornets","Chicago Bulls",
         "Cleveland Cavaliers","Dallas Mavericks","Denver Nuggets","Detroit Pistons","Golden State Warriors",
         "Houston Rockets","Indiana Pacers","LA Clippers","Los Angeles Lakers","Memphis Grizzlies",
         "Miami Heat","Milwaukee Bucks","Minnesota Timberwolves","New Orleans Pelicans","New York Knicks",
         "Oklahoma City Thunder","Orlando Magic","Philadelphia 76ers","Phoenix Suns","Portland Trail Blazers",
         "Sacramento Kings","San Antonio Spurs","Toronto Raptors","Utah Jazz","Washington Wizards"]

for team in teams:

    sentiment = teamSentiment(team)
    send_server = json.dumps({team:sentiment,"isTeam":True})
    
    if count < 14:
        sentiment = teamSentiment(team)
        send_server = json.dumps({team:sentiment})
        count += 1
    else:
        time.sleep(1000)
        sentiment = teamSentiment(team)
        send_server = json.dumps({team:sentiment})
        count = 0

    print(send_server)
    sys.stdout.flush()
    
#print("Completed!")

