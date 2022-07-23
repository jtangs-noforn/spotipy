import os
import re
import souptify.lyrics_finder as lyrics_finder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
import nltk.data
import nltk.corpus
from nltk.corpus import stopwords
import wordcloud 
from wordcloud import WordCloud, STOPWORDS
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download("stopwords")


def driver(flask_df):
    '''
    Given flask dataframe of last 50 songs, creates all outputs.

    Input:  
        flask_df (dataframe): last 50 songs from user
    
    Output: list of wordcloud png names
    '''
    print("in driver")
    df = make_df(flask_df)
    grouped = group_date(df)

    #scatter_emotions(grouped)
    scatter_emotions_50(df, grouped)
    grouped["representative"] = grouped.apply(lambda row: most_representative_song(df,row.name, row), axis  = 1)

    list_names = word_data(grouped)

    return list_names

def make_df(flask_df):
    '''
    Given flask df, adds sentiments, lyrics, and words to past 50 songs.

    Inputs:
        flask_df (dataframe): last 50 songs from user
    
    Output: dataframe with updated columns and lyrics.
    '''
    df = flask_df

    df["lyrics"] = df.apply(lambda row: lyrics_finder.get_lyrics(row["artistName"], row["trackName"]), axis = 1)
    df["words"] = df.apply(lambda row: lyrics_finder.get_relevant_words(row["lyrics"]), axis = 1) # should i have periods with this?

    print(type(df["msPlayed"][0]), df["msPlayed"][0])
    df["msPlayed"] = df.apply(lambda row: int(row["msPlayed"])/1000, axis = 1)
    df.rename(columns= {"msPlayed": "sPlayed"}, inplace = True)
    
    df["scores"] = df.apply(lambda row: add_emotions(row), axis = 1)
    df["negative"], df["neutral"], df["positive"], df["composite"] = zip(*df.scores)
    df.drop("scores", axis = 1, inplace = True)
    
    df["num_words"] = df.apply(lambda row: len(row["words"]), axis = 1)

    return df

def add_emotions(row):
    '''
    For a row in dataframe, call sentiment analyzer on the lyrics

    Input:
        row (Series): row in dataframe
    
    Output: tuple of negative, neutral, positive, and composite scores
    '''
    #print(row)
    sid = SentimentIntensityAnalyzer()
    #print(type(row["lyrics"]))
    dict_scores = sid.polarity_scores(row["lyrics"])

    return tuple(dict_scores.values())

def group_date(df):
    '''
    Given dataframe, group by date for all songs that have lyrics

    Input: 
        df (dataframe): dataframe with lyrics
    
    Output: grouped dataframe by date
    '''
    df.drop(df[df["words"].str.len() == 0].index, inplace = True)
    df = df.reset_index(drop = True)
    df["count"] = 1

    wa = lambda x: np.average(x, weights = df.loc[x.index, "sPlayed"])
   
    grouped = df.groupby("date").agg(all_words = ("words", "sum"), positive_w = ("positive", wa), \
    negative_w = ("negative", wa), count = ("count", "sum"), num_words_sum = ("num_words", "sum"), \
    num_words_mean = ("num_words", "mean"))

    return grouped


def scatter_emotions_50(df, grouped):
    '''
    Make scatterplot of positive and negative emotions
    for all 50 last songs.

    Inputs:
        df (dataframe): 50 songs with lyrics
        grouped (dataframe): songs grouped by date
    '''
    plt.clf()
    for index, row in df.iterrows():
        plt.scatter(row["positive"], row["negative"])
    
    for date, row in grouped.iterrows():
        plt.scatter(row["positive_w"], row["negative_w"], label = "average from {}".format(date))
    
    plt.legend(loc=2)
    plt.xlim([-0.05,0.7])
    plt.ylim([-0.05,0.7])
    plt.title("Lyrics Sentiments by Date")
    plt.xlabel('Positive Valence')
    plt.ylabel('Negative  Valence')
    # plt.show()
    plt.savefig("souptify/sentiments_by_song.png")

def word_cloud(dict_words, date):
    '''
    Create a wordcloud for a specific day. 

    Inputs: 
        dict_words (dictionary): get frequency of each word 
        date (string): day for wordcloud title and png name
    
    Output: name of png file created
    '''
    plt.clf()
    wc = WordCloud(width = 800, height = 800, background_color = "white",\
     max_font_size = 120, random_state = 15, max_words = 200)

    wc.generate_from_frequencies(dict_words)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud on " + date)
    # plt.show()
    figname = "wc" + date + ".png"
    plt.savefig("souptify/" + figname)

    return figname



def word_data(grouped):
    '''
    Makes dictionary mapping words to frequency and a wordcloud
    for each date. 

    Inputs:
        grouped (dataframe): 50 songs grouped by date
   
    Output: list of png names 
    '''
    all_words = grouped["all_words"].sum()
    unique_words = set(all_words)

    list_series = []
    list_names = []
    for date, row in grouped.iterrows():
        word, count = np.unique(row["all_words"], return_counts = True)

        dict_words = dict(zip(word, count))
        figname = word_cloud(dict_words, date)
        list_names.append(figname)

        ser = pd.Series(dict_words, name = date)

        list_series.append(ser)
    
    word_df = pd.concat(list_series, axis = 1)
    word_df = word_df.fillna(0).reset_index()
    return list_names

def most_representative_song(df, date, row):
    '''
    For a date, find most representative song listened to that day
    in terms of sentiment closeness. 

    Inputs:
        df (dataframe): list of 50 last songs
        date (string): date for representative song
        row (Series): row in grouped df

    Output: song_id for most representative song from that day
    '''
    best_id = None
    smallest_diff = None

    cut_df = df.loc[df["date"] == date]

    for _, song in cut_df.iterrows():
        diff = (song["positive"] - row["positive_w"])**2.0
        diff += (song["negative"] - row["negative_w"])**2.0

        if smallest_diff is None or diff < smallest_diff:
            smallest_diff = diff
            best_id = song["song_id"]
    
    return best_id
