# Importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from imblearn.over_sampling import SMOTE
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics 
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import oauth2
import os 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from souptify.primary_dataset_creator import getTrackFeatures

# File Paths
file_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = 'data'
final_dataset_path = os.path.join(file_dir, data_folder, 'final_dataset.csv')
df_final = pd.read_csv(final_dataset_path)

# Insert Spotify Creds
"""
cid = '189f473edf684b3499ac4b32de44a0ad'
secret = '5d1f62afa9b542838f6f5176f4df5208'
redirect_uri='http://localhost:7777/callback/'
username = 'mzuckerman'
scope = 'user-library-read user-read-recently-played user-top-read playlist-modify-private playlist-modify-public'
token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret, redirect_uri=redirect_uri)
sp = spotipy.Spotify(auth=token)
"""
file_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = 'data'
final_dataset_path = os.path.join(file_dir, data_folder, 'final_dataset.csv')


def getRecs(sp):
    '''
    Given Spotify user, generates recommended playlist per last 50 songs.

    Inputs: 
        sp (Spotify object): spotify object 
    '''
    df_fav = prepUserData(sp)
    
    df_final = pd.read_csv(final_dataset_path)

    df_original = PerformML(df_final, df_fav)
    profile = sp.me()
    username = profile['id']
    CreatePlaylist(sp, username, 'Fresh Waveforms', 'This playlist was created by the UglySoup team!')
    playlist_id = FetchPlaylists(sp,username)[0]

    list_tracks = df_original.loc[df_original["prediction"] == 1]["track_id"]
    
    EnrichPlaylist(sp, username, playlist_id, list_tracks)


def prepUserData(sp):
    '''
    Gets the last 50 songs of user in df for ML. 

    Inputs:
        sp (Spotify object): spotify object 

    Output: last 50 songs in a dataframe
    '''
    recently_played = sp.current_user_recently_played(limit=50)
    track_id = []
    for i, items in enumerate(recently_played['items']):
        track_id.append(items['track']['id'])

    df_favourite = pd.DataFrame({"track_id": track_id})

    fav_tracks = []
    for track in df_favourite['track_id']:
        try:  
            track = getTrackFeatures(sp, track)
            fav_tracks.append(track)
        except:
            pass

    df_fav = pd.DataFrame(fav_tracks, columns = ['track_id', 'name', 'album', 'artist', 'release_date', 'length', 'popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature'])
    df_fav = df_fav.drop(columns=['name', 'album', 'artist', 'release_date'])

    # Creating favorite column to use in classification
    df_fav['favorite'] = 1
    
    return df_fav


def PerformML(df_final, df_fav):
    '''
    Do machine learning to get most similar songs to df_fav.

    Inputs:
        df_final (dataframe): all songs from final_dataset.csv
        df_fav (dataframe): last 50 songs listened

    Output: dataframe with matches in final_dataset.csv 
    '''
    df = pd.concat([df_final, df_fav], axis=0)

    # Shuffle dataset.
    shuffle_df = df.sample(frac=1)

    # Define a size for your train set.
    train_size = int(0.8 * len(df))

    # Split dataset.
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]

    X = train_set.drop(columns=['favorite', 'track_id'])
    y = train_set.favorite

    # Train / Split Data
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X, y) 

    # Setting test datasets
    X_test = test_set.drop(columns=['favorite', 'track_id'])
    y_test = test_set['favorite']

    # Optimization for Decision Tree Classifier.
    parameters = {
        'max_depth':[3, 4, 5, 6, 10, 15,20, 30],
    }
    dtc = Pipeline([('CV',GridSearchCV(DecisionTreeClassifier(), parameters, cv = 5))])
    dtc.fit(X_train, y_train)
    dtc.named_steps['CV'].best_params_

    # Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth=30).fit(X_train, y_train)
    dt_scores = cross_val_score(dt, X_train, y_train, cv=10, scoring="f1")
    np.mean(dt_scores)

    pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=30))
    pipe.fit(X_train, y_train)  
    Pipeline(steps=[('standardscaler', StandardScaler()),
                ('dt', DecisionTreeClassifier(max_depth=30))])
    pipe.score(X_test, y_test)

    df_original = pd.read_csv(final_dataset_path)

    # Predicting if a song is a favorite.
    prob_preds = pipe.predict_proba(df_original.drop(['favorite','track_id'], axis=1))
    threshold = 0.30 
    preds = [1 if prob_preds[i][1]> threshold else 0 for i in range(len(prob_preds))]
    df_original['prediction'] = preds
    df_original['prediction'].value_counts()
    return df_original


# Function that builds a playlist in the user's Spotify account.
def CreatePlaylist(sp, username, playlist_name, playlist_description):
    playlists = sp.user_playlist_create(username, playlist_name, description = playlist_description)
    #return playlists

# Check if the playlist was created successfully.
def FetchPlaylists(sp, username):
    """
    Returns the user's playlists.
    """
        
    ids = []
    name = []
    num_tracks = []
    
    playlists = sp.user_playlists(username)
    for playlist in playlists['items']:
        ids.append(playlist['id'])
        name.append(playlist['name'])
        num_tracks.append(playlist['tracks']['total'])

    return ids
# Getting the playlist ID of the most recently made playlist so can add songs to it later.
#playlist_id = FetchPlaylists(sp,username)['id'][0]

# Add selected songs to playlist.
def EnrichPlaylist(sp, username, playlist_id, playlist_tracks):
    index = 0
    results = []
    
    while index < len(playlist_tracks):
        results += sp.user_playlist_add_tracks(username, playlist_id, tracks = playlist_tracks[index:index + 50])
        index += 50

# List of tracks to add.
#list_track = df.loc[df['prediction']  == 1]['track_id']
