import streamlit as st
import pandas as pd
import numpy as np
from streamlit_modal import Modal
import ast
import joblib
from PIL import Image


#***********************************************************************************
# configuration de la page
st.set_page_config(layout="wide")
 
# initialisation de la page si n'existe pas encore dans session_state
if "page" not in st.session_state:
    st.session_state["page"] = "accueil"

#***********************************************************************************
# chargement des modèles
knn = joblib.load("../BD_A_IGNORE/modele_knn.pkl")
scaler = joblib.load("../BD_A_IGNORE/scaler.pkl")
tfidf_vectorizer = joblib.load("../BD_A_IGNORE/tfidf_vectorizer.pkl")

# chargement des bases de films
@st.cache_data
def load_data():    
    try:
        df_movies_reco = joblib.load("../BD_A_IGNORE/df_movies.pkl")
        df_movies_now = joblib.load("../BD_A_IGNORE/df_now_playing.pkl")
        df_movies_future = joblib.load("../BD_A_IGNORE/df_upcoming_movie.pkl")

        return df_movies_reco, df_movies_now, df_movies_future

    except Exception as e:
        print(f"Erreur lors du chargement du fichier pickle: {e}")
        return None, None, None
    
df_movies_reco, df_movies_now, df_movies_future = load_data()

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
# paramètres barre de navigation et affichage des pages

st.markdown("""
<style>
.nav-bar {
    display: flex;
    align-items: center;  
    justify-content: space-between; 
    /*padding: 20px;*/
    /*border-radius: 20px;*/
}
.nav-link {
    padding: 10px 30px; /* Ajuster le padding pour une meilleure apparence */
    background-color:white;
    color: rgb(148, 73, 189) !important;
    border-radius: 16px;
    border: none;
    cursor: pointer;
    display: inline-block;
    font-weight: bold;
    text-align: center;
    text-decoration: none !important;
    font-size: 25px;
    margin: 4px 2px;
    box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.3);
    transition: background 0.3s;
}
.nav-link:hover {
    background-color:rgb(201, 183, 211);
}
.logo-h1 { 
    margin-right: 0px
}
.logo-text { 
    font-size: 60px;
    font-weight: bold;
    color:black;
    font-family: Arial, sans-serif;
}

</style>
""", unsafe_allow_html=True)

# Barre de navigation (dans un seul conteneur)
logo_path = "static\logo.PNG"
st.markdown("""
<div class="nav-bar">
    <div class="logo-h1">
        <img src={logo_path} alt="Votre Logo" style="height: 100px; margin-right: 30px;">
    </div>
    <div class="nav-buttons">  <a href="?page=accueil" class="nav-link">Accueil</a>
        <a href="?page=recommandation" class="nav-link">Application</a>
            <a href="?page=analyse" class="nav-link">Indicateurs clés films</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Gestion des paramètres et affichage des pages
params = st.query_params
current_page = params.get("page", "accueil")

if current_page == "accueil":
    st.title("Bienvenue sur What Movie 2D?")
    #image = Image.open("Capture.JPG")
    #st.image(image, caption="Image de présentation", use_column_width=True)
    st.markdown("""
    <img src="static/Capture.JPG" style="float: left" alt="Image" style="width:100%;"/>
    <div style='font-size:18px;'>
    <br>Vous hésitez sur quel film regarder ce soir ?
    <br> <b>What Movie 2D?</b>  est une plateforme conçue pour vous aider à découvrir des films
    qui correspondent à vos goûts grâce à une recommandation personnalisée basée sur les données.<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:18px;'>
    <br> <b> Pourquoi cette plateforme ?</b>
    <br>Ce projet a été conçu dans le cadre de mon parcours en Data Analysis afin de mettre en pratique mes compétences en collecte, <br>
    traitement et analyse de données, tout en explorant l’univers du Machine Learning appliqué aux recommandations.<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:18px;'>
    <br> <b> Que trouverez-vous ici ?</b>
    <br>* <b>Application de recommandation</b> : Entrez vos préférences et laissez l'algorithme vous suggérer des films adaptés<br>
    traitement et analyse de données, tout en explorant l’univers du Machine Learning appliqué aux recommandations.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:18px;'>
    <br> * <b>Indicateurs d'analyse </b> : Visualisez des statistiques sur les films collectés, explorez les tendances du cinéma et comprenez les mécanismes derrière les recommandations.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
                <div style="
                font-weight: bold;
                border: 2px solid #04AA6D;
                box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19);
                padding: 10px 20px;
                text-align: center;
                cursor: pointer;
                background-color: #f0f8ff;
            ">
            Je veux une recommandation de films
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
                <div style="
                font-weight: bold;
                border: 2px;
                box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19);
                padding: 10px 20px;
                text-align: center;
                cursor: pointer;
                background-color: #f0f8ff;
            ">
            Je veux voir les tendances du cinéma
        </div>
        """, unsafe_allow_html=True)

elif current_page == "recommandation":
    st.title("Bienvenue sur l'application de recommandation de films What Movie 2D?")

elif current_page == "analyse":
    st.title("Bienvenue, ici explorez les tendances du cinéma.")
else:
    st.write("Page non trouvée")