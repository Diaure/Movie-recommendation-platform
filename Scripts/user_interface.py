import streamlit as st
import pandas as pd
import joblib
import os
import subprocess
import ast
from streamlit_modal import Modal
import schedule
import time


#***********************************************************************************
# configuration de la page
st.set_page_config(page_title="Recommandation de Films", page_icon="🎬", layout="wide")
 
# initialisation de la page si n'existe pas encore dans session_state
if "page" not in st.session_state:
    st.session_state["page"] = "accueil"

#***********************************************************************************
# CHARGEMENT DES MODELES & DONNEES STATIQUES
#***********************************************************************************

#chargement modèles
knn = joblib.load("../BD_A_IGNORE/modele_knn.pkl")
scaler = joblib.load("../BD_A_IGNORE/scaler.pkl")
tfidf_vectorizer = joblib.load("../BD_A_IGNORE/tfidf_vectorizer.pkl")

# chargement données fixes
@st.cache_data
def load_static_data():    
    try:
        df_movies = joblib.load("../BD_A_IGNORE/df_movies.pkl")
        return df_movies

    except Exception as e:
        print(f"Erreur lors du chargement du fichier pickle: {e}")
        return None, None, None
    
df_movies = load_static_data()

#***********************************************************************************
# TRAITEMENT, CHARGEMENT DES DONNEEs ("actuellement au cinéma", "à venir")
#***********************************************************************************
# vérifier la dernière mise à jour
def check_update():
    if os.path.exists("last_update.txt"):
        with open("last_update.txt", "r") as f:
            return f.read()
    return ""

last_update = check_update()


# # charger le fichier de traitement des données "traitement.py" (si les fichiers n'existent pas)
# @st.cache_data
# def run_treatment_script():
#     try:
#         files = ["../BD_A_IGNORE/df_now_playing.pkl", "../BD_A_IGNORE/df_upcoming_movie.pkl"]
#         if not all(os.path.exists(f) for f in files):
#             with st.spinner("Mise à jour des films en cours... (Cela peut prendre plusieurs minutes)"):
#                 result = subprocess.run(["python", "traitement.py"], capture_output=True, text=True)
#                 if result.returncode != 0:
#                     st.error(f"Erreur dans traitement.py : {result.stderr}")
#                     return False
#         return True
#     except Exception as e:
#         st.error(f"Échec de l'exécution de traitement.py : {e}")
#         return False

# chargement des données après la mise à jour planifiée
@st.cache_data(ttl=3600, hash_funcs={str: lambda _: last_update})
def load_updated_data():
    try:
        df_now_playing = joblib.load("../BD_A_IGNORE/df_now_playing.pkl")
        df_upcoming = joblib.load("../BD_A_IGNORE/df_upcoming_movie.pkl")

        # vérifier et remplirldes valeurs manquantes
        df_now_playing.fillna("", inplace=True)
        df_upcoming.fillna("", inplace=True)

        return df_now_playing, df_upcoming
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des fichiers mis à jour : {e}")
        return None, None

df_now_playing, df_future = load_updated_data()

#***********************************************************************************
# RECUPERATION DES AFFICHES SUR TMDB
#***********************************************************************************

# fonction pour récupérer le poster des films
def get_image(selected_movie):
    racine = 'https://image.tmdb.org/t/p/w600_and_h900_bestv2'
    try:
        poster_url = racine + df_movies[df_movies['originalTitle'] == selected_movie]['poster_path'].iloc[0]
        return poster_url
    except IndexError:
        print(f"Image non trouvée pour le film {selected_movie}")
        return "logo.PNG"

#***********************************************************************************
# FONCTION D'AFFICHAGE DU FILM SELECTIONNE PAR L'UTILISATEUR
#***********************************************************************************
 
def user_movie_choice(df_movies): 

    # création d'une colonne en minuscule pour faciliter les recherches
    df_movies['originalTitle_lower'] = df_movies['originalTitle'].str.lower()

    col1, col2, col3 = st.columns(3)

    with col1:
        # champ de saisie de l'utilisateur
        user_input = st.text_input("Saisissez un titre de film, puis tapez entrer :").lower()

    # filtrer les titres de films en fonction de la saisie utilisateur
    if user_input:
        filtered_movies = df_movies[df_movies['originalTitle_lower'].str.contains(user_input, na=False)]
        movie_list = filtered_movies['originalTitle'].tolist()

        if not movie_list:  # Si aucune correspondance, afficher un message
            st.warning("Aucun film trouvé avec ce titre.")

        with col2:
            # Selectbox pour afficher les suggestions
            selected_movie = st.selectbox("Choisissez le film correspondant à votre recherche :", movie_list)
            if selected_movie:
    
                # définir la table pour récupérer les informations du film
                movie_info = df_movies[df_movies['originalTitle'] == selected_movie]
                if not movie_info.empty:

                    # etraire et afficher des informations du film
                    overview = movie_info['overview'].iloc[0] 

                    # s'assurer que les genres sont bien des listes
                    movie_info['genres_liste'] = movie_info['genres_liste'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x) 
                    genres_str = ', '.join(movie_info['genres_liste'].iloc[0])

                    # convertir les colonnes acteurs et ordering en listes réelles
                    # s'assurer que les acteurs sont bien des listes
                    # movie_info['actors_name'] = movie_info['actors_name'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                    # acteurs = ', '.join(movie_info['actors_name'].iloc[0])  # Liste des acteurs

                    actors_data = movie_info['actors_name'].iloc[0]
                    if isinstance(actors_data, str):
                        actors = ast.literal_eval(actors_data)
                    else:
                        actors = actors_data
                    

                    ordering_data = movie_info['actors_rank'].iloc[0]
                    if isinstance(ordering_data, str):
                        ordering = list(map(int, ast.literal_eval(ordering_data)))
                    else:
                        ordering = list(map(int, ordering_data))
                    
                    
                    # movie_info['actors_rank'] = movie_info['actors_rank'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                    # movie_info['actors_rank'] = movie_info['actors_rank'].apply(lambda x: list(map(int, x)) if isinstance(x, list) else x)
                    # ordering = ', '.join(map, str, movie_info['actors_rank'].iloc[0])  # Liste des rangs

                    # extraire les acteurs principaux et secondaires
                    actors_with_rank = list(zip(actors, ordering))
                    actors_with_rank.sort(key=lambda x: x[1])  # Trie par ordre croissant de rang
                    actors_sorted = []
                    for actor, rank in actors_with_rank:
                        actors_sorted.append(actor)

                    actors_str = ', '.join(actors_sorted)
                            
                    # act_princip = main_actors[0] if main_actors else "Non disponible"
        
                    # extraire et sélectionner le réalisateur
                    movie_info['directors_name'] = movie_info['directors_name'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                    realisateur_str = ', '.join((movie_info['directors_name'].iloc[0])) 

                    # extraire et sélectionner le scenariste
                    movie_info['writers_name'] = movie_info['writers_name'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                    scenariste_str = ', '.join((movie_info['writers_name'].iloc[0]))

                    # extraire et sélectionner le producteur
                    movie_info['producers_name'] = movie_info['producers_name'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                    producteur_str = ', '.join((movie_info['producers_name'].iloc[0]))

                    # extraire et sélectionner le cinéaste
                    movie_info['cinematographers_name'] = movie_info['cinematographers_name'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                    cineaste_str = ', '.join((movie_info['cinematographers_name'].iloc[0]))

                    # extraire et sélectionner l'éditeur
                    movie_info['editors_name'] = movie_info['editors_name'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                    editeur_str = ', '.join((movie_info['editors_name'].iloc[0]))

                    # extraire et sélectionner la note
                    note = movie_info['averageRating'].iloc[0]  

                    # extraire et sélectionner l'année
                    movie_info['startYear'] = (pd.to_datetime(movie_info['startYear'])).dt.year
                    year = movie_info['startYear'].iloc[0]

                    # extraire et sélectionner la durée
                    time = int(movie_info['runtimeMinutes'].iloc[0])


                    with col3:
                        modal = Modal("Détails du film", key="film_modal", max_width=800)
                        if st.button("Voir les détails du film"):
                            st.markdown("""
                                <div style="
                                font-weight: bold;
                                border: 2px solid #04AA6D;
                                box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19);
                                padding: 10px 20px;
                                text-align: center;
                                cursor: pointer;
                                background-color: #f0f8ff;
                                </div>
                                """, unsafe_allow_html=True)
                            modal.open()

                        poster_url = get_image(selected_movie)

                        if modal.is_open():
                            with modal.container():
                                st.markdown(# afficher le poster et les informations du film
                                f"""
                                <div style="display: flex; align-items: space-between;">
                                <!-- Image du film -->
                                <img src="{poster_url}" style="margin-right: 10px; width:380px; height:420px;">
                                                <div style="max-width: 800px;">
                                                    <p style="margin: 0;"><strong> Synopsis :</strong> <em> {overview} </em></p>
                                                    <p style="margin: 0;"><strong> Réalisation par :</strong> {realisateur_str}</p>
                                                    <p style="margin: 0;"><strong> Genres :</strong> {genres_str}</p>
                                                    <p style="margin: 0;"><strong> Distribution :</strong> {actors_str}</p>                                       
                                                    <p style="margin: 0;"><strong> Ecrit par :</strong> {scenariste_str}</p>
                                                    <p style="margin: 0;"><strong>{year} ({time} minutes)</strong></p>
                                                    <p style="margin: 0;"><font size="16">{note}</font></p>
                                                </div>
                                </div>
                                """, 
                                        unsafe_allow_html=True)
                if movie_info.empty:
                    st.warning("Aucune information trouvée pour ce film.") 

#***********************************************************************************
# BARRE DE NAVIGATION
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

#***********************************************************************************
# Barre de navigation (dans un seul conteneur)
logo_path = "static/Capture_2.PNG"
st.markdown("""
<div class="nav-bar">
    <div class="logo-h1">
        <img src="static/Capture_2.PNG" alt="Votre Logo" style="height: 100px; margin-right: 30px;">
    </div>
    <div class="nav-buttons">  <a href="?page=accueil" class="nav-link">Accueil</a>
        <a href="?page=recommandation" class="nav-link">Application</a>
            <a href="?page=analyse" class="nav-link">Indicateurs clés films</a>
    </div>
</div>
""", unsafe_allow_html=True)

#***********************************************************************************
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
    st.title("Bienvenue sur l'application de recommandation de films")

    st.markdown("""
    <img src="static/Capture.JPG" style="float: left" alt="Image" style="width:100%;"/> </div>""", unsafe_allow_html=True)
    
    df_movies = load_static_data()
    user_movie_choice(df_movies)

elif current_page == "analyse":
    st.title("Bienvenue, ici explorez les tendances du cinéma.")
else:
    st.write("Page non trouvée")