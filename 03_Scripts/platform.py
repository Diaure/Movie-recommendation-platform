import streamlit as st
import pandas as pd
import numpy as np
from streamlit_modal import Modal
import ast
import joblib


#***********************************************************************************
# configuration de la page
st.set_page_config(layout="wide")
 
# Initialisation de la session.  vérifier si les clés sub_page et selected_movie sont absentes dans les données utilisateurs (st.session_tate)
if "sub_page" not in st.session_state:
    st.session_state["sub_page"] = "recommandations"  # Par défaut, nous initialisons la clé "sub_page", si elle n'existe pas, comme étant la page de recommandations
if "selected_movie" not in st.session_state:
    st.session_state["selected_movie"] = None # Par défaut, nous initialisons la clé "selected_movie", si elle n'existe pas, comme étant None

# Page de l'interface de l'utilisateur
st.sidebar.title("Navigation") # titre de la barre latérale
menu = st.sidebar.radio("", ["Home", "Application"]) # définition des menus de la barre latérale

#***********************************************************************************
# chargement des modèles
knn = joblib.load("../BD_A_IGNORE/modele_knn.pkl")
scaler = joblib.load("../BD_A_IGNORE/scaler.pkl")
tfidf_vectorizer = joblib.load("../BD_A_IGNORE/tfidf_vectorizer.pkl")

# chargement des bases de films
df_movies_reco = pd.read_pickle("../BD_A_IGNORE/df_movies.pkl")
df_movies_now = pd.read_pickle("../BD_A_IGNORE/df_movies_now.pkl")
df_movies_future = pd.read_pickle("../BD_A_IGNORE/df_movies_future.pkl")

#***********************************************************************************
# fonction pour récupérer le poster des films
def get_image(selected_movie):
    racine = 'https://image.tmdb.org/t/p/w600_and_h900_bestv2'
    try:
        poster_url = racine + df_movies_reco[df_movies_reco['originalTitle'] == selected_movie]['poster_path'].iloc[0]
        return poster_url
    except IndexError:
        print(f"Image non trouvée pour le film {selected_movie}")
        return None

#***********************************************************************************