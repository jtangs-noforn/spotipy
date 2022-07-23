from bs4 import BeautifulSoup
import requests
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import os 
import re
import string
import nltk
import nltk.corpus
from wordcloud import STOPWORDS
from nltk.corpus import stopwords
nltk.download("stopwords")


def get_lyrics(artist, title):
    '''
    Given artist and title of song, find corresponding lyrics on Genius.com

    Inputs:
        artist (string): name of artist
        title (string): name of song

    Output: string of lyrics
    '''
    token = "d2wGrbeTeqear0CyyGM9MqAClI72m56Oi915dXOcJ_vQfgqA7HLkzc74K2tANvGt"
    home_url = "https://api.genius.com"
    headers = {"Authorization": "Bearer " + token}

    artist, title = clean_name(artist, title)

    data = {'q': artist + " " + title}

    search_out = requests.get(home_url + "/search?", params= data, headers = headers)

    json_search = search_out.json()

    song_url = ""
    print(title, artist)
    for song in json_search['response']['hits']:
        print(song["result"]["title"].lower().replace("\u200b", ""), song["result"]["primary_artist"]["name"])
        if song['result']['title'].lower().replace("\u200b", "") == title and artist in song['result']['primary_artist']['name'].lower():
            song_url = song['result']['url']
            break
    
    print(song_url)
    
    if song_url != "":
        lyrics_request = requests.get(song_url)
        soup = BeautifulSoup(lyrics_request.text, "html.parser")
        
        text_obj = soup.find("div", class_ = "SongPage__LyricsWrapper-sc-19xhmoi-5 UKjRP")
        words = clean_text(text_obj)
        return words
    else:
        print("Cannot find {} by {}".format(title, artist))
        return ""

def clean_name(artist, title):
    '''
    Clean the name of artist and title of song

    Inputs: 
        artist (string): name of artist
        title (string): name of song
    
    Output: cleaned tuple of artist and title
    '''
    artist = artist.lower()
    title = title.lower()
    title = re.sub(r' \([^)]*\)', "", title)
    title = title.replace("'", "’")
    title = title.split(" - ")[0]

    return (artist, title)

def clean_text(text_object):
    '''
    Cleans text object into lyrics that can be analyzed.

    Inputs:
        text_object (object): object to make into text
    
    Output: clean string
    '''
    text = ""
    
    if text_object is not None:
        text = text_object.get_text(separator = " ")
        if "Embed" in text:
            text = text.split("Embed", 1)[0]
        if "Lyrics" in text:
            text = text.split("Lyrics", 1)[1]
        clean = re.sub(r'[\(\[].*?[\)\]]',"", text)
        clean = re.sub(" +", " ", clean)
        clean = re.sub(r'[' + string.punctuation + '\d]', "", clean).strip()
    
        return clean
    else:
        print("Cannot find tag")
        return ""

def get_relevant_words(lyrics):
    '''
    Turn string into list of relevant words.

    Inputs:
        lyrics (string): lyrics of song
    
    Output: list of relevant words
    '''
    words = lyrics.split(" ")

    stopwords_own = [strip_punctuation(word) for word in stopwords.words("english")]
    stopwords_own.append("")
    stopwords_own.extend([strip_punctuation(word) for word in STOPWORDS if word not in stopwords.words("english")])

    words = [word.lower() for word in words if word.lower() not in stopwords_own]
    return words

def strip_punctuation(word):
    '''
    Get rid of any punctuation in a word

    Inputs:
        word (string): word

    Output: word without punctuation
    '''
    return re.sub(r'['+ string.punctuation + ']', "", word)





