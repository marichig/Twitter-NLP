import tweepy
from tweepy import TweepError
import pandas as pd
import numpy as np
import json
import re
import pprint
import datetime, time
import pickle

class TweetCollector:
    def __init__(self,
    consumer_key,
    consumer_secret,
    access_key,
    access_secret,
                 *,
    environment_name_30_day = None,
    environment_names_full = None,
    rate_lim_standard = (180, 900, 100),
    rate_lim_full = (30, 60, 100),
    rate_lim_30_day = (30, 60, 100),
    output_folder = ""
    ):
        '''
        Need to provide your own Twitter Developer Credentials.

        Environment names: if you use the 30-day or full archive mode (either in the sandbox or paid mode), need to provide the enviroment names you chose.

        Rate limits and max tweets per request depend on your tier of access. They are currently set to the free level of access (i.e., sandbox for 30 day or full).
        If you have paid for a higher tier of access, you can pass the rate limits for your tier to "rate_lim_standard", "rate_lim_full", or "rate_lim_30_day".
        Pass the rate limits as a triple in the form (number of calls/time unit, time unit in seconds, max tweets per request). For instance, the time unit
        for the 30 day and full tier is 60 seconds, but for the standard tier it is 900 seconds (15 minutes). The rate limits for the premium tier can be found at:
        https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/overview/premium

        For understanding the Tweet object structure, see https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/overview/intro-to-tweet-json
        '''
        self.CONSUMER_KEY = consumer_key
        self.CONSUMER_SECRET = consumer_secret
        self.ACCESS_KEY = access_key
        self.ACCESS_SECRET = access_secret

        #Set these when performing Premium search
        self.environment_name_30_day = environment_name_30_day
        self.environment_names_full = environment_names_full

        self.standard_operators = "-filter:retweets lang:en"
        self.premium_operators = "-is:retweet lang:en"
        self.search_operators = {"standard": self.standard_operators, "30 day": self.premium_operators, "full": self.premium_operators}

        self.auth = tweepy.AppAuthHandler(self.CONSUMER_KEY, self.CONSUMER_SECRET)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

        self.rate_limits = {
            #name: (number of calls/time unit, time unit in seconds, max tweets per request)
            "standard": rate_lim_standard,
            "30 day": rate_lim_full,
            "full": rate_lim_30_day
                      }
        self.output_folder = output_folder

        self.results = None
        self.logs = None


    def set_search_operators(self, standard = None, premium = None):
        '''
        Pass operators as a string to be included in the search query.

        Standard operators (available using the free endpoint for past seven day search) are available here:
        https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/guides/standard-operators

        Premium operators (available when using the paid tier, or the sandbox testing environment) are available:
        https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/guides/premium-operators
        '''
        __set_standard_operators(standard)
        __set_premium_operators(premium)

    def __set_standard_operators(self, standard = None):
        if standard is not None:
            self.standard_operators = standard
            self.search_operators["standard"] = self.standard_operators # I think this isn't necessary? "standard" is always pointing to the instance variable

    def __set_premium_operators(self, premium = None):
        if premium is not None:
            self.premium_operators = premium
            self.search_operators["30 day"] = self.premium_operators # I think this isn't necessary? "standard" is always pointing to the instance variable
            self.search_operators["full"] = self.premium_operators


    class DummyTweet:
        '''
        Inner class used for testing reverse chronological collection.
        '''
        def random_timestamp(self, start, end):
            x = np.random.normal(1,0.0002)
            return datetime.datetime.fromtimestamp(
                max(
                    (1 - abs(1-x)) * end.timestamp(),
                    start.timestamp()
                ),
                pytz.utc)

        def __init__(self, start, end, text):
            dt = self.random_timestamp(start, end)
            self.timestamp = dt.strftime("%a %b %d %H:%M:%S %z %Y")
            self.text = text
            self._json = {'created_at': self.timestamp, 'text': self.text}


    def perform_search(self, query, start_time_object, end_time_object, mode, test = False):
        if test:
            print("TEST MODE:")
            print("Searching from "
                            + start_time_object.strftime('%Y-%m-%d-%H-%M') + " GMT to "
                            + end_time_object.strftime('%Y-%m-%d-%H-%M')
                            + " GMT for query: " + query)
           # print("standard query:",query + " until:" + str(int(end_time_object.timestamp()))
           #             + " since:"+ str(int(start_time_object.timestamp())))
           # print("premium query:", query)
            return [self.DummyTweet(start_time_object, end_time_object, "TEST")
                    for _ in range(min(500, int(np.random.normal(loc = 700, scale = 100))))
                   ]

        if mode == "standard":
            return self.api.search(
                    q=query + " until:" + str(int(end_time_object.timestamp()))
                        + " since:"+ str(int(start_time_object.timestamp())),
                    tweet_mode = "extended",
                    lang = "en",
                    count = self.rate_limits["standard"][2]
                )
        if mode == "full":
            return self.api.search_full_archive(
                    self.environment_names_full,
                    query=query,
                    fromDate = start_time_object.strftime('%Y%m%d%H%M'),
                    toDate = end_time_object.strftime('%Y%m%d%H%M'),
                    maxResults = self.rate_limits["full"][2]
                )
        if mode == "30 day":
            return self.api.search_30_day(
                    self.environment_name_30_day,
                    query=query,
                    fromDate = start_time_object.strftime('%Y%m%d%H%M'),
                    toDate = end_time_object.strftime('%Y%m%d%H%M'),
                    maxResults = self.rate_limits["30 day"][2] #max tweets / request
                )

        raise Exception("""Unsupported Search Mode!
        Must be one of \"standard\", \"full\", or \"30 day\", case and space sensitive.""")

    def protect_rate_limit(self, call_counter, rate_tuple):
        call_cap_per_unit_time = rate_tuple[0]
        call_unit_time = rate_tuple[1]
        if call_counter % call_cap_per_unit_time == 0 and call_counter != 0:
                        print("Sleeping for",str(call_unit_time + 15),"seconds to avoid rate limit")
                        time.sleep(call_unit_time + 15)

    def reverse_chronological_collection(self, queries, start_date_object, end_date_object, mode = "standard"):
        query_to_results_map = {query:[] for query in queries}
        logs = []
        call_counter = 0

        for query in queries:
            print("Reverse collecting for", query)
            end_time = end_date_object
            while start_date_object < end_time:

                self.protect_rate_limit(call_counter, self.rate_limits[mode])

                try:
                    results = self.perform_search(
                        " ".join([query, self.search_operators[mode]]),
                        start_date_object,
                        end_time,
                        mode,
                        #test = True
                    )

                    timestamps = [
                        datetime.datetime.strptime(tweet._json['created_at'],"%a %b %d %H:%M:%S %z %Y")
                            for tweet in results]
                    timestamps.sort()

                    earliest_timestamp = start_date_object if len(timestamps) == 0 else timestamps[0]

                    logs.append("Collected " + str(len(results)) + " from "
                            + earliest_timestamp.strftime('%Y-%m-%d-%H-%M') + " GMT to "
                            + end_time.strftime('%Y-%m-%d-%H-%M')
                            + " GMT for query: " + query)
                    query_to_results_map[query].extend(results)

                    end_time = earliest_timestamp


                except Exception as e:
                    print("Uh-oh, error:")
                    print(e)
                    logs.append("ERROR raised for search "
                            + start_date_object.strftime('%Y-%m-%d-%H-%M') + " GMT to "
                            + end_time.strftime('%Y-%m-%d-%H-%M')
                            + " GMT for query: " + query)
                    title = "BACK_UP_data_" + start_date_object.strftime('%m-%d') + "-"+ end_time.strftime('%m-%d')
                    self.results = query_to_results_map
                    self.logs = logs

                    pickle.dump(query_to_results_map, open(self.output_folder+ title + ".pkl", "wb"))
                    pickle.dump(logs, open(self.output_folder+ "logs_"+title+".txt", "wb"))
                    return query_to_results_map, logs


                print(logs[-1])
                time.sleep(1)
                call_counter+=1
                #if len(results) < self.rate_limits[mode][2]-1: #sometimes 499 are returned for instance
                    #print("We should be done! Did not hit the rate limit --> we exhausted tweets through this range.")
                    #break
                    #only possible if we've gone through the start date

        print("Exiting ~~~")
        self.results = query_to_results_map
        self.logs = logs
        return query_to_results_map, logs

    def save_tweets_jsons(self, path_name = None, results = None):
        '''
        Exports tweet data to json, using native tweet json format..

        :param path_name: Path name for exporting the json. Don't include the file extension
        :param results: Dictionary of results to write.
        If you only want to write results for a certain query,
        you can pass that to results as long as it's a dict.
        '''
        if results is None:
            results = self.results
        if path_name is None:
            path_name = self.output_folder
        tweet_jsons = []
        for query in results:
            tweet_jsons.extend([tweet._json for tweet in results[query]])

        with open(path_name +'.json', 'w', encoding='utf-8') as f:
            json.dump(tweet_jsons, f, ensure_ascii=False, indent=4)


    def save_logs(self, path_name = None):
        if path_name is None:
            path_name = self.output_folder
        pickle.dump(self.logs, open(path_name + ".txt", "wb"))
