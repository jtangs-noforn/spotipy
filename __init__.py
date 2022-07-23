import os

from flask import Flask

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, url_for, session, request, redirect, send_file, render_template
import json
import time
import pandas as pd
import numpy as np
import souptify.emotions as emotions
from datetime import date
import souptify.user_dataset_creator_ML
from souptify.user_dataset_creator_ML import getRecs

SPOTIPY_CLIENT_ID='c3a9aa8b8a644921886ab4c3f1157c3a'
SPOTIPY_CLIENT_SECRET='20b691ebb0414e6093043cde391dc87c' 

# App config
def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    
    app.secret_key = 'sjdsaodjal'
    app.config['SESSION_COOKIE_NAME'] = 'spotify-login-session'

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def login():
        '''
        Login page for spotify authorization.
        '''
        sp_oauth = create_spotify_oauth()
        auth_url = sp_oauth.get_authorize_url()
        print("authorize")
        return redirect(auth_url)

    @app.route('/redirect')
    def redirectPage():
        print("at redirect")
        sp_oauth = create_spotify_oauth()
        session.clear()
        code = request.args.get('code')
        token_info = sp_oauth.get_access_token(code, check_cache=False)
        session["token_info"] = token_info
        return redirect(url_for('getTracks', _external=True))

    @app.route('/logout')
    def logout():
        for key in list(session.keys()):
            session.pop(key)
        return redirect('/')

    @app.route('/getTracks')
    def getTracks():
        try:
            token_info = get_token()
        except:
            print('user not logged in')
            redirect('/')

        sp = spotipy.Spotify(auth=token_info['access_token'])
        recents = sp.current_user_recently_played(limit=50)
        profile = sp.me()
        user_id = profile['id']
        redirect('/logout')

        song_lst = []

        for item in recents['items']:
            #print(item)
            print("\n\n")
            track = item['track']
            artist = track['artists'][0]['name']
            artist_id = track['artists'][0]['id']
            album = track['album']['name']
            album_id = track['album']['id']
            song = track['name']
            song_id = track['id']
            timing = item['played_at']
            date = timing.split("T", 1)[0]
            endTime = timing.split("T", 1)[1]
            duration = int(track["duration_ms"])
            print(type(duration), duration)

            song_lst.append([artist, artist_id, album, album_id, 
                            song, song_id, date, endTime, duration]) 

        labels = ['artistName', 'artist_id', 'album', 'album_id',   
                'trackName', 'song_id', 'date', "endTime", "msPlayed"]

        df = pd.DataFrame(np.array(song_lst), columns = labels)

        global list_images
        list_images = emotions.driver(df)
        print(list_images)

        print("where did we get")
        getRecs(sp)

        return render_template('index.html')


    # creates app routes for max of 5 word clouds
    @app.route('/get_word_cloud1/')
    def get_word_cloud1():
        '''
        Retrieves first word cloud, if it exists.
        '''
        filename = get_filename(1)
        return send_file(filename, mimetype="image/png")
        #<img src="localhost:5000/get_word_cloud"/>

    @app.route('/get_word_cloud2/')
    def get_word_cloud2():
        '''
        Retrieves second word cloud, if it exists.
        '''
        filename = get_filename(2)
        return send_file(filename, mimetype="image/png")

    @app.route('/get_word_cloud3/')
    def get_word_cloud3():
        '''
        Retrieves third word cloud, if it exists.
        '''
        filename = get_filename(3)
        print(filename)
        return send_file(filename, mimetype="image/png")

    @app.route('/get_word_cloud4/')
    def get_word_cloud4():
        '''
        Retrieves fourth word cloud, if it exists.
        '''
        filename = get_filename(4)
        print(filename)
        return send_file(filename, mimetype="image/png")

    @app.route('/get_word_cloud5/')
    def get_word_cloud5():
        '''
        Retrieves fifth word cloud, if it exists.
        '''
        filename = get_filename(5)
        print(filename)
        return send_file(filename, mimetype="image/png")

    #Creates route for lyrics mood chart
    @app.route('/get_mood_plot/')
    def get_mood_plot():
        '''
        Retrieves image for lyric moods.
        '''
        filename = "sentiments_by_song.png"
        return send_file(filename, mimetype="image/png")

    return app

app = create_app(test_config=None)

# Checks to see if token is valid and gets a new token if not
def get_token():
    token_info = session.get('token_info', None)
    '''
    if not token_info:
        raise 'exception'
    now = int(time.time())
    is_token_expired = token_info['expires_at'] - now < 60
    if (is_token_expired):
    '''
    sp_oauth = create_spotify_oauth()
    token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
    session.clear()
    return token_info #token_valid


def create_spotify_oauth():
    '''
    Helper function for getting spotify data
    '''
    return SpotifyOAuth(
            client_id='c3a9aa8b8a644921886ab4c3f1157c3a',
            client_secret='20b691ebb0414e6093043cde391dc87c' ,
            redirect_uri=url_for('redirectPage', _external=True),
            scope='user-library-read user-read-recently-played user-top-read playlist-modify-private playlist-modify-public')
            
            #"user-library-read user-read-currently-playing user-read-recently-played")

def get_filename(i):
    '''
    Helper function to retrieve the filename of a certain word cloud 
    (up to a max of 5), given a wordcloud number. Returns the file (if it exists)
    or a default blank file, if there is no word cloud.
    '''
    if len(list_images) >= i:
        filename = list_images[i - 1]
    else:
        filename = "1x1.png"
    return filename


