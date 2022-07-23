# Importing required libraries
import time
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
from functools import reduce
import spotipy
import os
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import oauth2
'''
file_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = 'data'
# Insert Spotify Creds
cid = '189f473edf684b3499ac4b32de44a0ad'
secret = '5d1f62afa9b542838f6f5176f4df5208'
redirect_uri='http://localhost:7777/callback/'
token = util.prompt_for_user_token(client_id=cid, client_secret=secret, redirect_uri=redirect_uri)
sp = spotipy.Spotify(auth=token)
'''
# Get features of each track.
def getTrackFeatures(sp, track_id):
  meta = sp.track(track_id)
  features = sp.audio_features(track_id)

  # meta
  track_id = track_id
  name = meta['name']
  album = meta['album']['name']
  artist = meta['album']['artists'][0]['name']
  release_date = meta['album']['release_date']
  length = meta['duration_ms']
  popularity = meta['popularity']

  # features
  acousticness = features[0]['acousticness']
  danceability = features[0]['danceability']
  energy = features[0]['energy']
  instrumentalness = features[0]['instrumentalness']
  liveness = features[0]['liveness']
  loudness = features[0]['loudness']
  speechiness = features[0]['speechiness']
  tempo = features[0]['tempo']
  time_signature = features[0]['time_signature']

  track = [track_id, name, album, artist, release_date, length, popularity, danceability, acousticness, energy, instrumentalness, liveness, loudness, speechiness, tempo, time_signature]
  return track
