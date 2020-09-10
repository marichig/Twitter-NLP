import pandas as pd
import json
import pickle

class TweetDataConverter:
    def __init__(self,
    tweet_jsons,
    status_attr = ['full_text',
               'created_at',
               'id_str',
               'quoted_status_id_str',
               'quoted_status',
               'in_reply_to_status_id_str',
               'in_reply_to_user_id_str',
               'retweet_count',
               'favorite_count',
               'entities',
               'place',
               'coordinates'],
    user_attr = ['id_str',
             'name',
             "description",
             "location",
             "verified",
             "followers_count",
             "friends_count",
             "listed_count",
             "favourites_count",
             "statuses_count"]):
        self.tweet_jsons = tweet_jsons
        self.status_attr = status_attr
        self.user_attr = user_attr

        self.data_dict = {} #map from attributes --> values array, n-th index is n-th tweet
        self.dataframe = None
        self.encounteredIDs = set()
        self.is_data_dict_filled = False

    def is_retweet(self, status):
        return "retweeted_status" in status.keys()
    
    def initialize_dict(self):
        if not self.data_dict:
            for attr in self.status_attr:
                self.data_dict['status.' + attr] = []
            for attr in self.user_attr:
                self.data_dict['author.' + attr] = []
                
                
    def write_tweet_to_data(self, tweet):
        self.initialize_dict()
        for attr in self.status_attr:
            if attr not in ['quoted_status_id_str', 'quoted_status', 'full_text', 'entities']: #handle elsewhere
                self.data_dict['status.' + attr].append(tweet[attr])
        for attr in self.user_attr:
            if attr in tweet['user']:
                self.data_dict['author.' + attr].append(tweet['user'][attr])

        if not tweet['truncated']: ## checked that all with truncated false do not have a extented_tweet object.
            self.data_dict['status.full_text'].append(tweet['full_text'])
            self.data_dict['status.entities'].append(tweet['entities'])
        else:
            self.data_dict['status.full_text'].append(tweet['extended_tweet']['full_text'])
            self.data_dict['status.entities'].append(tweet['extended_tweet']['entities'])

        #manually handle not-always-present-fields
        if tweet['is_quote_status']:
            try: # Deleted quoted tweets have an ID but not the tweet object
                self.data_dict["status.quoted_status"].append(tweet["quoted_status"])
                self.data_dict["status.quoted_status_id_str"].append(tweet["quoted_status_id_str"])
            except:
                self.data_dict["status.quoted_status_id_str"].append(None)
                self.data_dict["status.quoted_status"].append(None)
        else:
            self.data_dict["status.quoted_status_id_str"].append(None)
            self.data_dict["status.quoted_status"].append(None)

    def fill_data_dict(self):
        num_duplicates = 0
        for tweet in self.tweet_jsons:
            #to avoid duplicate tweets:
            if tweet['id_str'] not in self.encounteredIDs:
                 self.encounteredIDs.add(tweet['id_str'])
            else:
                num_duplicates+=1
                continue

            if self.is_retweet(tweet):
                print("Retweets found!")
                num_duplicates+=1

            else:
                self.write_tweet_to_data(tweet)
        print("number of duplicate tweets =", num_duplicates)
        self.is_data_dict_filled = True


    def to_dataframe(self, sort = True, sort_by_column = 'status.created_at'):
        if not self.is_data_dict_filled:
            self.fill_data_dict()

        if not self.dataframe:
            df = pd.DataFrame(self.data_dict)
            if 'status.created_at' in df.columns:
                df['status.created_at'] =pd.to_datetime(df['status.created_at'])

            if sort:
                df = (df.sort_values(by=sort_by_column))
                df.reset_index(drop = True, inplace = True)

            self.dataframe = df

        return self.dataframe
