SOUPTIFY

*** Prep to run the program ***

Please clear your internet/chrome/browser cache :)
Navigate to the interface folder (user:~/soupgroup/interface$)

Then run these commands in the terminal in specified order:
    pip install spotipy --upgrade
    pip install wordcloud
    pip install nltk
    pip install matplotlib
    pip install imblearn
    You may need to restart your kernel or VSCode 
    and then run the following lines
    
    pip install flask --upgrade
    export FLASK_APP=souptify
    flask run

Files: 
    1. emotions.py
    2. lyrics_finder.py
    3. __init__.py
    4. user_dataset_creator_ML.py
