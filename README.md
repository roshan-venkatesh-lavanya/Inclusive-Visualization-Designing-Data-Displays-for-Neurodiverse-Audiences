Overview

The Inclusive Emotional Movie Explorer is an interactive visualization tool that maps movies based on emotional similarity. It combines the MovieLens dataset with the NRC-VAD Lexicon, which provides Valence, Arousal, and Dominance scores for English words. By matching movie tags with VAD values, the app computes an emotional profile for each film and visualizes similarity using a radial network.

Built using Python, Plotly, and Dash, the app supports ADHD- and autism-friendly design principles such as simplified modes, calming color gradients, and reduced visual load.

Features

Radial layout of emotionally similar movies

Node color = emotional valence

Node size = audience engagement (mean rating)

Hover tooltips with emotional & genre details

Click any node to recenter the network

Toggle between Simplified and Intense view modes

NRC-VAD–driven emotional modeling

Fully interactive Dash web app

Tech Stack

Python

Pandas / NumPy

Scikit-Learn

Plotly

Dash

MovieLens Dataset

NRC-VAD Lexicon

How to Run Locally
pip install -r requirements.txt
python app.py


App will launch at:

http://127.0.0.1:8050

Deployment Notes (Render)

Create a Web Service and use:

Start Command:

gunicorn app:app.server


Include these files in the repo:

app.py

requirements.txt

Procfile (optional but recommended)

data files (movies.csv, tags.csv, ratings.csv, NRC-VAD-Lexicon.txt)

License

This project uses the publicly available MovieLens dataset and NRC-VAD lexicon under their respective licenses.
