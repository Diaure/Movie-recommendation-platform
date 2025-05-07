import streamlit as st
import pandas as pd
import joblib
import os
import subprocess
import ast
from streamlit_modal import Modal
from PIL import Image
from datetime import datetime
from .scraping import fetch_movies_with_credits, fetch_upcoming_movies_with_credits
from scipy.sparse import hstack
import pickle
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import re
import plotly_express as px


#***********************************************************************************
# configuration de la page
st.set_page_config(page_title="Recommandation de Films", page_icon="🎬", layout="wide")

# Image de fond (cinéma)
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0.5, 0.5, 0.5, 0.5), rgba(0.5, 0.5, 0.5, 0.5)),
                        url("https://images.unsplash.com/photo-1542204165-65bf26472b9b?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OHx8ZmlsbXxlbnwwfHwwfHx8MA%3D%3D");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.65);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
 
# initialisation de la page si n'existe pas encore dans session_state
if "page" not in st.session_state:
    st.session_state["page"] = "accueil"

#***********************************************************************************
# CHARGEMENT DES MODELES & DONNEES STATIQUES
#***********************************************************************************

#chargement des features et des modèles

with open("../BD_A_IGNORE/features_list.pkl", "rb") as f:
    features = pickle.load(f)

knn = joblib.load("../BD_A_IGNORE/modele_knn.pkl")
scaler = joblib.load("../BD_A_IGNORE/scaler.pkl")
tfidf_vectorizer = joblib.load("../BD_A_IGNORE/tfidf_vectorizer.pkl")

#***********************************************************************************
# FONCTION DE CHARGEMENT DES DONNEES STATIQUES
#***********************************************************************************

# chargement données fixes
@st.cache_data
def load_static_data():    
    try:
        df_movies = joblib.load("../BD_A_IGNORE/df_movies.pkl")
        return df_movies

    except Exception as e:
        print(f"Erreur lors du chargement du fichier pickle: {e}")
        return None

# charger les données
df_movies = load_static_data()

#***********************************************************************************
# MISE A JOUR
#*********************************************************************************** 
def check_update():
    if os.path.exists("last_update.txt"):
        with open("last_update.txt", "r") as f:
            return f.read()
    return ""

last_update = check_update()

# scraping des films tmdb
def update_movie_data():
    try:
        with st.spinner("🔄 Scraping des films en cours..."):
            df_now_playing = fetch_movies_with_credits()
            df_upcoming = fetch_upcoming_movies_with_credits()

            os.makedirs("../BD_A_IGNORE", exist_ok=True)
            joblib.dump(df_now_playing, "../BD_A_IGNORE/df_now_playing.pkl")
            joblib.dump(df_upcoming, "../BD_A_IGNORE/df_upcoming_movie.pkl")

            # Mettre à jour le fichier "last_update.txt"
            with open("last_update.txt", "w") as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return True
    except Exception as e:
        st.error(f"❌ Échec de la mise à jour via scraping : {e}")
        return False

def run_treatment_script():
    try:  
        result = subprocess.run(["python", "traitement.py"], capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Erreur traitement.py : {result.stderr}")
            return False
        return True
    except Exception as e:
        st.error(f"Échec exécution traitement.py : {e}")
        return False
    
# Auto-scraping + traitement si fichiers manquants
def ensure_data_ready():
    required_files = [
        "../BD_A_IGNORE/df_now_playing.pkl",
        "../BD_A_IGNORE/df_upcoming_movie.pkl"
    ]
    if not all(os.path.exists(f) for f in required_files):
        st.warning("🔎 Fichiers manquants : lancement automatique du scraping et traitement.")
        if update_movie_data() and run_treatment_script():
            st.success("✅ Données mises à jour automatiquement.")
        else:
            st.error("❌ Échec mise à jour automatique.")

ensure_data_ready()
    
#***********************************************************************************
# CHARGEMENT DES DONNEES TRAITEES
#*********************************************************************************** 
@st.cache_data(ttl=3600, hash_funcs={str: lambda _: last_update})
def load_updated_data():
    try:
        df_now_playing = joblib.load("../BD_A_IGNORE/df_now_playing.pkl")
        df_upcoming = joblib.load("../BD_A_IGNORE/df_upcoming_movie.pkl")
        df_now_playing.fillna("", inplace=True)
        df_upcoming.fillna("", inplace=True)
        return df_now_playing, df_upcoming
    except Exception as e:
        st.error(f"Erreur chargement données mises à jour : {e}")
        return None, None

df_now_playing, df_upcoming = load_updated_data()

#***********************************************************************************
# ZONE ADMIN AFIN DE FORCER LA MISE A JOUR
#***********************************************************************************
# Ajout d'un mode admin pour autoriser la mise à jour manuelle uniquement si l'admin est activé
is_admin = st.secrets["admin"]["mode"].lower() == "true"

if is_admin:
    with st.expander("🔐 Zone Admin : Forcer la mise à jour des films"):
        st.success("Mode admin activé")
        if st.button("🔄 Forcer la mise à jour maintenant"):
            if update_movie_data() and run_treatment_script():
                st.success("✅ Mise à jour réussie.")
                st.rerun()
            else:
                st.error("❌ Échec de la mise à jour.")

#***********************************************************************************
# RECUPERATION DES AFFICHES SUR TMDB
#***********************************************************************************

# fonction pour récupérer le poster des films
def get_image(selected_movie, df_source):
    racine = 'https://image.tmdb.org/t/p/w600_and_h900_bestv2'
    try:
        poster_path = df_source[df_source['originalTitle'] == selected_movie]['poster_path'].iloc[0]
        if pd.isna(poster_path) or poster_path == '':
            raise ValueError("Aucune image disponible")
        return racine + poster_path  
    except (IndexError, ValueError) as e:
        print(f"Image non trouvée pour le film '{selected_movie}' : {e}")
        return "https://via.placeholder.com/300x450?text=No+Image"

#***********************************************************************************
# FONCTION D'AFFICHAGE DES DETAILS DU FILM SELECTIONNE PAR L'UTILISATEUR
#***********************************************************************************

def user_choice(df_movies):
    # création d'une colonne en minuscule pour faciliter les recherches
    df_movies['originalTitle_lower'] = df_movies['originalTitle'].str.lower()

    col1, col2= st.columns(2)

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
                selected_movie = st.selectbox("Sélectionnez le film correspondant à votre recherche :", movie_list)
                if selected_movie:
                    st.session_state["selected_movie"] = selected_movie

# fonction détails des films
def movie_details(df, selected_movie):
                        
    # définir la table pour récupérer les informations du film
    movie_info = df[df['originalTitle'] == selected_movie]
    if not movie_info.empty:

        # etraire et afficher des informations du film
        overview = movie_info['overview'].iloc[0] 

        # s'assurer que les genres sont bien des listes
        movie_info['genres_liste'] = movie_info['genres_liste'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x) 
        genres_str = ', '.join(movie_info['genres_liste'].iloc[0])

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

        # extraire les acteurs et leur rang
        actors_with_rank = list(zip(actors, ordering))
        actors_with_rank.sort(key=lambda x: x[1])  # Trie par ordre croissant de rang
        actors_sorted = []
        for actor, rank in actors_with_rank:
            actors_sorted.append(actor)
        if actors_sorted:
            st.session_state["main_actor"] = actors_sorted[0]  # acteur principal
        else:
            st.session_state["main_actor"] = None

        actors_str = ', '.join(actors_sorted)

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
        try:
            start_year_raw = movie_info['startYear'].iloc[0]
            
            # Gérer les types chaîne comme "\N", NaN, etc.
            if isinstance(start_year_raw, str):
                if start_year_raw.strip().lower() in ["\\n", "nan", ""]:
                    year = "Année inconnue"
                else:
                    year = int(start_year_raw)
            elif pd.isna(start_year_raw):
                year = "Année inconnue"
            else:
                year = int(start_year_raw)
        except:
            year = "Année inconnue"

        # extraire et sélectionner la durée
        time = int(movie_info['runtimeMinutes'].iloc[0])

        poster_url = get_image(selected_movie, df_movies)
        st.markdown(f""" 
        <div style="display: flex; align-items: space-between;">
                <!-- Image du film -->
                <img src="{poster_url}" style="margin-right: 10px; width:620px; height:400px;">
                <div style="max-width: 1000px;">
                    <p style="margin: 0;"><strong> Synopsis :</strong> <em> {overview} </em></p>
                    <p style="margin: 0;"><strong> Année de sortie : {year}</strong></p>
                    <p style="margin: 0;"><strong> Durée : {time} minutes</strong></p>
                    <p style="margin: 0;"><strong> Réalisation :</strong> {realisateur_str}</p>
                    <p style="margin: 0;"><strong> Genres :</strong> {genres_str}</p>
                    <p style="margin: 0;"><strong> Distribution :</strong> {actors_str}</p> 
                    <p style="margin: 0;"><strong> Scénario :</strong> {scenariste_str}</p> 
                    <p style="margin: 0;"><strong> Production :</strong> {producteur_str}</p>   
                    <p style="margin: 0;"><strong> Cinématographie :</strong> {cineaste_str}</p> 
                    <p style="margin: 0;"><strong> Montage :</strong> {editeur_str}</p>                                    
                    <p style="margin: 0;"> <strong> <font size="12">{movie_info['averageRating'].iloc[0]}/10</strong></p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    if movie_info.empty:
        st.warning("Aucune information trouvée pour ce film.")

#***********************************************************************************
# AFFICHAGE DES RECOMMANDATIONS
#***********************************************************************************
df_now_playing = df_now_playing.rename(columns={'vote_average': 'averageRating', 'vote_count': 'numVotes'})
df_upcoming = df_upcoming.rename(columns={'vote_average': 'averageRating', 'vote_count': 'numVotes'})

def align_features(df_input, features_reference):
    for col in features_reference:
        if col not in df_input.columns:
            df_input[col] = 0
    return df_input[features_reference]

# ---- fonction de recommandation ----
def recommander_films(film_titre, df, knn, scaler, tfidf_vectorizer, n_recommendations=10):

    # vérifier si le film est bien dans la base
    if film_titre not in df['originalTitle'].values:
        return "Film non trouvé dans la base."

    try:
        features = [col for col in ['runtimeMinutes', 'averageRating', 'numVotes', 'popularity', 'budget'] if col in df.columns]
        features += [col for col in df.columns if col.startswith('genre_')]

        # Trouver l’index du film dans df_movies
        film_index_movies = df[df['originalTitle'] == film_titre].index[0]

        # Extraire ses caractéristiques et standardiser
        film_features = df.loc[film_index_movies, features].values.reshape(1, -1)
        film_features_scaled = scaler.transform(film_features)

        # Transformer `overview` en vecteur TF-IDF
        film_overview = df.loc[film_index_movies, 'overview']
        film_overview_tfidf = tfidf_vectorizer.transform([film_overview])

        # Fusionner les caractéristiques numériques et le TF-IDF
        film_vector = hstack([film_features_scaled, film_overview_tfidf])

        # Trouver les films les plus proches
        distances, indices = knn.kneighbors(film_vector)

        # Récupérer les films recommandés
        recommandations = df.iloc[indices[0][1:]]  # Exclure le film lui-même

        # Retourner les films avec plus d'infos (titre, période, genres, rating, popularité)
        return recommandations[['originalTitle', 'periode', 'averageRating', 'popularity'] + [col for col in df_movies.columns if col.startswith('genre_')]]

    except Exception as e:
        return f"Erreur lors de la recommandation : {e}"

# ---- fonction de recommandation permettant d'entrainer les modèles sur les nouvelles bases ----
def recommander_depuis_autre_df(film_titre, df_source, df_target, knn, scaler, tfidf_vectorizer, features, n_recommendations=10):
    try:
        match = df_source[df_source['originalTitle'] == film_titre]
        if match.empty:
            return f"Film '{film_titre}' non trouvé dans df_source"

        index = match.index[0]

        # Features numériques
        X_features = df_source.loc[[index]]
        X_features = align_features(X_features, features)
        X_scaled = pd.DataFrame(scaler.transform(X_features), columns=features)
        # X_num = scaler.transform(df_source.loc[[index], features].reindex(columns=features))

        # Texte (overview)
        overview = df_source.loc[index, 'overview']
        X_text = tfidf_vectorizer.transform([overview])

        # Fusion
        vector_film = hstack([X_scaled, X_text])

        # Aligner df_target à df_source
        df_target_aligned = align_features(df_target.copy(), features)
        X_target_num = scaler.transform(df_target_aligned)
        X_target_text = tfidf_vectorizer.transform(df_target["overview"].fillna(""))
        X_target_full = hstack([X_target_num, X_target_text])

        missing_features_source = [feat for feat in features if feat not in df_source.columns]
        missing_features_target = [feat for feat in features if feat not in df_target.columns]

        # Prédiction avec le knn entraîné sur df_target
        knn.fit(X_target_full)
        distances, indices = knn.kneighbors(vector_film)

        # On exclut le film lui-même si présent
        max_index = len(df_target) - 1
        valid_indices = [i for i in indices[0] if i <= max_index and df_target.iloc[i]['originalTitle'] != film_titre]
        reco_indices = valid_indices[:n_recommendations]

        return df_target.iloc[reco_indices]

    except Exception as e:
        return f"Erreur lors de la recommandation : {e}"

# ---- fonction pour la recommandation par acteur ----
def recommander_par_acteur(acteur, df, n=10):
    films = []
    for _, row in df.iterrows():
        actors = row["actors_name"]
        ranks = row["actors_rank"]

        if isinstance(actors, list) and acteur in actors:
            idx = actors.index(acteur)
            rank = ranks[idx] if isinstance(ranks, list) and idx < len(ranks) else 999
            films.append((row, rank))

    # On trie les films par rang de l'acteur dans le film (plus petit = plus important)
    sorted_films = sorted(films, key=lambda x: x[1])
    
    # On récupère les lignes (sans le rang)
    return pd.DataFrame([film[0] for film in sorted_films[:n]])


# ---- fonction d'affichage des recommandations ----
def recommendation_show():
    selected_movie = st.session_state.get("selected_movie", None)

    if not selected_movie:
        st.warning("Veuillez sélectionner un film d'abord.")
        return

    # films dans le même genre
    st.subheader(" 🎬 Films dans le même genre")
    movie_reco = recommander_films(film_titre=selected_movie, df=df_movies, knn=knn, scaler=scaler, tfidf_vectorizer=tfidf_vectorizer, n_recommendations=10)

    if isinstance(movie_reco, str):
        st.error(movie_reco)
        return
    
    cols = st.columns(len(movie_reco))
    for i, col in enumerate(cols):
        with col:
            title = movie_reco["originalTitle"].iloc[i]
            st.image(get_image(title, df_movies), use_container_width=True)

            if st.button(f"{title}", key=f"genre_modal_btn_{i}"):
                st.session_state[f"open_modal_{i}"] = True
            #st.caption(title)

        if st.session_state.get(f"open_modal_{i}", False):
            modal = Modal(title, key=f"modal_genre_{i}", max_width=1000)
            with modal.container():
                movie_details(df_movies, title)
                if st.button("Fermer", key=f"close_genre_modal_{i}"):
                    st.session_state[f"open_modal_{i}"] = False

    # films avec l'acteur principal
    act_princip = st.session_state.get("main_actor", None)
    if act_princip:
        st.subheader(f"Films avec {act_princip}")
        movie_main_actor = recommander_par_acteur(act_princip, df_movies)
        
        cols_actor = st.columns(len(movie_main_actor))
        for i, col in enumerate(cols_actor):
            with col:
                actor_title = movie_main_actor["originalTitle"].iloc[i]
                st.image(get_image(actor_title, df_movies), use_container_width=True)
                #st.caption(actor_title)

                if st.button(f"{actor_title}", key=f"actor_modal_btn_{i}"):
                    st.session_state[f"open_modal_actor{i}"] = True

        if st.session_state.get(f"open_modal_actor{i}", False):
            modal_actor = Modal(actor_title, key=f"modal_actor_{i}", max_width=1000)
            with modal_actor.container():
                movie_details(df_movies, actor_title)
    else:
        st.info("Acteur principal non disponible.")

    # films actuellement au cinéma
    st.subheader("Films actuellement au cinéma")
    movie_now_playing = recommander_depuis_autre_df(film_titre=selected_movie, df_source=df_movies, df_target=df_now_playing, knn=knn, scaler=scaler, tfidf_vectorizer=tfidf_vectorizer, features=features, n_recommendations=10)

    if isinstance(movie_now_playing, pd.DataFrame) and not movie_now_playing.empty:
        num_movies = movie_now_playing.shape[0]
     
        cols_now = st.columns(num_movies)
        for i, col in enumerate(cols_now):
            with col:
                now_title = movie_now_playing["originalTitle"].iloc[i]
                st.image(get_image(now_title, df_now_playing), use_container_width=True)
                # st.caption(movie_now_playing["originalTitle"].iloc[i])

                if st.button(f"{now_title}", key=f"now_modal_btn_{i}"):
                    st.session_state[f"open_modal_now{i}"] = True

        if st.session_state.get(f"open_modal_now{i}", False):
            modal_now = Modal(now_title, key=f"modal_now_{i}", max_width=1000)
            with modal_now.container():
                movie_details(df_now_playing, now_title)
    else:
        st.info("Aucun film à recommander pour les séances en cours.")

    # films bientôt au cinéma
    st.subheader("Films prochainement au cinéma")
    movie_upcoming = recommander_depuis_autre_df(film_titre=selected_movie, df_source=df_movies, df_target=df_upcoming, knn=knn, scaler=scaler, tfidf_vectorizer=tfidf_vectorizer, features=features, n_recommendations=10)
    
    if isinstance(movie_upcoming, pd.DataFrame) and not movie_upcoming.empty:
        num_umovies = movie_upcoming.shape[0]

        cols_up = st.columns(num_umovies)
        for i, col in enumerate(cols_up):
            with col:
                upcoming_title = movie_upcoming["originalTitle"].iloc[i]
                st.image(get_image(upcoming_title, df_upcoming), use_container_width=True)
                # st.caption(movie_upcoming["originalTitle"].iloc[i])

                if st.button(f"{upcoming_title}", key=f"upc_modal_btn_{i}"):
                    st.session_state[f"open_modal_upc{i}"] = True

        if st.session_state.get(f"open_modal_upc{i}", False):
            modal_upc = Modal(upcoming_title, key=f"modal_upc_{i}", max_width=1000)
            with modal_upc.container():
                movie_details(df_upcoming, upcoming_title)
    else:
        st.info("Aucun film à recommander pour les séances à venir.")       

#***********************************************************************************
# ANALYSE DES FILMS
#***********************************************************************************

# ---- fonction pour afficher les filtres ----
def show_kpis(df):
    
    df['startYear_clean'] = pd.to_numeric(df['startYear'], errors='coerce')

    # extraire les pays de production uniques: transformation en liste réelle - créer une nouvelle colonne - extraire les valeurs uniques dans un set 
    country_dict = {'BA': 'Bosnie-Herzégovine', 'LR': 'Libéria', 'IQ': 'Irak', 'AM': 'Arménie',
    'FI': 'Finlande', 'ID': 'Indonésie', 'RW': 'Rwanda', 'GT': 'Guatemala', 'PL': 'Pologne', 'AZ': 'Azerbaïdjan',
    'UY': 'Uruguay', 'AU': 'Australie', 'YU': 'Yougoslavie', 'CM': 'Cameroun', 'BS': 'Bahamas', 'IT': 'Italie',
    'CH': 'Suisse', 'SV': 'Salvador', 'AO': 'Angola', 'UA': 'Ukraine', 'CV': 'Cap-Vert', 'MU': 'Maurice',
    'KE': 'Kenya', 'EC': 'Équateur', 'KH': 'Cambodge', 'XG': 'Guernesey', 'LV': 'Lettonie', 'TW': 'Taïwan',
    'YE': 'Yémen', 'SK': 'Slovaquie', 'BN': 'Brunei', 'MK': 'Macédoine du Nord', 'LY': 'Libye', 'RU': 'Russie',
    'AN': 'Antilles néerlandaises','EE': 'Estonie', 'IN': 'Inde', 'PE': 'Pérou', 'AL': 'Albanie', 'GR': 'Grèce',
    'BE': 'Belgique', 'ZM': 'Zambie', 'SU': 'Union soviétique', 'SN': 'Sénégal', 'MW': 'Malawi', 'SL': 'Sierra Leone',
    'GD': 'Grenade', 'XK': 'Kosovo','BZ': 'Belize','CY': 'Chypre','CO': 'Colombie','MY': 'Malaisie','HT': 'Haïti',
    'BR': 'Brésil','GP': 'Guadeloupe','CR': 'Costa Rica','SG': 'Singapour','SB': 'Îles Salomon','VN': 'Vietnam',
    'JO': 'Jordanie','RO': 'Roumanie','CD': 'République démocratique du Congo','TR': 'Turquie','MC': 'Monaco',
    'MQ': 'Martinique','GI': 'Gibraltar','FO': 'Îles Féroé','US': 'États-Unis','NI': 'Nicaragua','DJ': 'Djibouti',
    'ML': 'Mali','NZ': 'Nouvelle-Zélande','VU': 'Vanuatu','FR': 'France','PM': 'Saint-Pierre-et-Miquelon',
    'BF': 'Burkina Faso','GL': 'Groenland','NL': 'Pays-Bas','CF': 'République centrafricaine','GW': 'Guinée-Bissau',
    'KW': 'Koweït','HU': 'Hongrie','PH': 'Philippines','SA': 'Arabie saoudite','GE': 'Géorgie','UZ': 'Ouzbékistan',
    'JP': 'Japon','CZ': 'République tchèque','EG': 'Égypte','BH': 'Bahreïn','GM': 'Gambie','KR': 'Corée du Sud',
    'GA': 'Gabon','GB': 'Royaume-Uni','DE': 'Allemagne','DO': 'République dominicaine','BY': 'Biélorussie',
    'KG': 'Kirghizistan','CL': 'Chili','NO': 'Norvège','IS': 'Islande','PS': 'Palestine','TZ': 'Tanzanie','MT': 'Malte',
    'WS': 'Samoa','MZ': 'Mozambique','LT': 'Lituanie','AQ': 'Antarctique','BM': 'Bermudes','LA': 'Laos',
    'PT': 'Portugal','IR': 'Iran','PR': 'Porto Rico','CS': 'Serbie-Monténégro','MX': 'Mexique','BD': 'Bangladesh',
    'PG': 'Papouasie-Nouvelle-Guinée','NA': 'Namibie','MG': 'Madagascar','GH': 'Ghana','KM': 'Comores','MD': 'Moldavie',
    'SE': 'Suède','ZA': 'Afrique du Sud','VG': 'Îles Vierges britanniques','HK': 'Hong Kong','IL': 'Israël',
    'SI': 'Slovénie','AT': 'Autriche','TL': 'Timor oriental','CA': 'Canada','NP': 'Népal','LI': 'Liechtenstein',
    'BI': 'Burundi','ZW': 'Zimbabwe','XC': 'Île Christmas','SR': 'Suriname','KY': 'Îles Caïmans','CU': 'Cuba',
    'TD': 'Tchad','BB': 'Barbade','TH': 'Thaïlande','RS': 'Serbie','SO': 'Somalie','SZ': 'Eswatini','VE': 'Venezuela',
    'ET': 'Éthiopie','MN': 'Mongolie','BJ': 'Bénin','ER': 'Érythrée','PK': 'Pakistan','FJ': 'Fidji','MA': 'Maroc',
    'TJ': 'Tadjikistan','QA': 'Qatar','CG': 'République du Congo','AW': 'Aruba','NG': 'Nigeria','BG': 'Bulgarie',
    'BT': 'Bhoutan','LU': 'Luxembourg','FK': 'Îles Falkland','GQ': 'Guinée équatoriale','TT': 'Trinité-et-Tobago',
    'MO': 'Macao','CX': 'Île Christmas','BO': 'Bolivie','LB': 'Liban','SY': 'Syrie','PF': 'Polynésie française',
    'DK': 'Danemark','SD': 'Soudan','LK': 'Sri Lanka','AI': 'Anguilla','ES': 'Espagne','PY': 'Paraguay',
    'KP': 'Corée du Nord','AF': 'Afghanistan','GN': 'Guinée','HN': 'Honduras','MV': 'Maldives','AE': 'Émirats arabes unis',
    'CI': "Côte d'Ivoire",'XI': 'Île Christmas','AD': 'Andorre','TC': 'Îles Turques-et-Caïques','JM': 'Jamaïque',
    'DZ': 'Algérie','IE': 'Irlande','MM': 'Myanmar','NC': 'Nouvelle-Calédonie','MR': 'Mauritanie','HR': 'Croatie',
    'UG': 'Ouganda','KZ': 'Kazakhstan','SJ': 'Svalbard et Jan Mayen','CN': 'Chine','NE': 'Niger','GY': 'Guyana',
    'LS': 'Lesotho','BW': 'Botswana','AR': 'Argentine','TN': 'Tunisie','VA': 'Vatican','PA': 'Panama','ME': 'Monténégro'}
    
    def map_country_list(country_list_str):
        try:
            country_codes = ast.literal_eval(country_list_str)
            return [country_dict.get(code, code) for code in country_codes]
        except (ValueError, TypeError):
            return []

    df["prod_country_list"] = df["production_countries"].apply(map_country_list)

    bins = [1, 3, 5, 7, 9, 10.1] # les bornes
    labels = ["1 à 2", "3 à 4", "5 à 6", "7 à 8", "9 à 10"] # groupes
    df["note_group"] = pd.cut(df["averageRating"], bins=bins, labels=labels, include_lowest=True) # right=False <=> bornes gauches incluses
    
     # ------------ Filtres ------------
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        with col1:
            decade_labels = sorted(df['periode'].dropna().unique().tolist())
            decade_options = ["Toutes"] + decade_labels
            selected_decade = st.multiselect("Filtrer par décennie", decade_options, default="Toutes")

        with col2:
            availble_years = sorted(df['startYear_clean'].dropna().unique().astype(int).tolist())
            year_options = ["Toutes"] + availble_years
            selected_years = st.selectbox("Année de sortie", year_options)

        with col3:
            all_genres_set = set()
            for sublist in df['genres_liste']:
                if isinstance(sublist, list):
                    all_genres_set.update(sublist)
            all_genres = sorted(all_genres_set)
            genre_options = ["Tous"] + all_genres
            selected_genres = st.multiselect("Genres", genre_options, default="Tous")

        with col4:
            available_notes = sorted(df['note_group'].dropna().unique().tolist())
            notes_options = ["Toutes"] + available_notes 
            selected_note_group = st.multiselect("Notes (1 à 10):", notes_options, default="Toutes")

        with col5:
            all_countries = set()
            for countries in df['prod_country_list']:
                all_countries.update(countries)
            country_options = sorted(list(all_countries))
            countries = ["Tous"] + country_options
            selected_country = st.multiselect("Choisir les pays de production :", countries, default="Tous")
    
    filtered_df = df.copy()         
                    
    # ------------ Application des filtres ------------
    if "Toutes" not in selected_decade and selected_decade:
        filtered_df = filtered_df[filtered_df['periode'].isin(selected_decade)]
        
    if selected_years != "Toutes":
        filtered_df = filtered_df[filtered_df['startYear_clean'] == int(selected_years)]

    if "Tous" not in selected_genres and selected_genres:
        filtered_df = filtered_df[filtered_df['genres_liste'].isin(selected_genres)]

    if "Toutes" not in selected_note_group and selected_note_group:
        filtered_df = filtered_df[filtered_df['note_group'].isin(selected_note_group)]

    if "Tous" not in selected_country and selected_country:
        filtered_df = filtered_df[filtered_df['prod_country_list'].apply(lambda x: any(c in x for c in selected_country))]
        

    # ------------ Afficher les kpis ------------
    col = st.columns((1, 4, 2.5), gap='medium')
    with col[0]:
        st.markdown('#### Indicateurs Clés')

        st.markdown(f"""<div style='background-color: #000000; border: 4px solid #ffffff; border-radius: 10px; padding: 5px; font-size: 26px; text-align: center;'> Total films<br><b>{len(filtered_df)}</b></div>""", unsafe_allow_html=True)
        st.markdown("<br>" \
        "<br>", unsafe_allow_html=True)
        
        all_genres = [genre for sublist in filtered_df['genres_liste'] for genre in sublist if isinstance(sublist, list)]
        unique_language = filtered_df['original_language'].unique()
        language_counts = len(unique_language)
        st.markdown(f"""<div style='background-color: #000000; border: 4px solid #ffffff; border-radius: 10px; padding: 5px; font-size: 26px; text-align: center;'> Langues<br><b>{language_counts}</b></div>""", unsafe_allow_html=True)
        st.markdown("<br>" \
        "<br>", unsafe_allow_html=True)
        
        time_average = round(filtered_df['runtimeMinutes'].mean(), 2)
        st.markdown(f"""<div style='background-color: #000000; border: 4px solid #ffffff; border-radius: 10px; padding: 5px; font-size: 26px; text-align: center;'> Durée moyenne<br><b>{time_average} (min)</b></div>""", unsafe_allow_html=True)
        st.markdown("<br>" \
        "<br>", unsafe_allow_html=True)
        
        note_average = round(filtered_df['averageRating'].mean(), 2)
        st.markdown(f"""<div style='background-color: #000000; border: 4px solid #ffffff; border-radius: 10px; padding: 5px; font-size: 26px; text-align: center;'> Note moyenne<br><b>{note_average}</b></div>""", unsafe_allow_html=True)
        st.markdown("<br>" \
        "<br>", unsafe_allow_html=True)

        popularity_average = round(filtered_df['popularity'].mean(), 2)
        st.markdown(f"""<div style='background-color: #000000; border: 4px solid #ffffff; border-radius: 10px; padding: 5px; font-size: 26px; text-align: center;'> Popularité moyenne<br><b>{popularity_average}</b></div>""", unsafe_allow_html=True)
        st.markdown("<br>" \
        "<br>", unsafe_allow_html=True)

        all_actors = [actor for sublist in filtered_df['actors_name'] for actor in sublist if isinstance(sublist, list)]
        actor_counts = Counter(all_actors)
        unique_actors = len(actor_counts) 
        st.markdown(f"""<div style='background-color: #000000; border: 4px solid #ffffff; border-radius: 10px; padding: 5px; font-size: 26px; text-align: center;'> Total acteurs<br><b>{unique_actors}</b></div>""", unsafe_allow_html=True)

        

    # ------------ Afficher les graphes ------------
    with col[1]:
        # Evolution du nombre de films par an
        st.markdown('#### Evolution du nombre de films produit par an')
        movie_1927 = filtered_df[filtered_df['startYear_clean'] >= 1927]
        movie_by_yr = movie_1927['startYear_clean'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(18, 6))
        sns.barplot(x=movie_by_yr.index, y=movie_by_yr.values, color='skyblue', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel('')
        st.pyplot(fig)

        # Genres les plus présents
        st.markdown('#### Genres les plus présents (Top 10)')
        top_genres = (
            filtered_df['genres_liste']
            .explode()
            .value_counts()
            .head(10)
        )
        fig1, ax = plt.subplots(figsize=(12, 4))
        sns.barplot(x=top_genres.index, y=top_genres.values, order=top_genres.index, palette='pastel', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel('')
        st.pyplot(fig1)

        # Evolution des durées moyennes des films par an
        st.markdown('#### Evolution des durées moyennes par an')
        movie_by_yr_time = movie_1927.groupby('startYear_clean')['runtimeMinutes'].mean().sort_index()
        fig2, ax = plt.subplots(figsize=(18, 5))
        sns.barplot(x=movie_by_yr_time.index, y=movie_by_yr_time.values, color='skyblue', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel('')
        st.pyplot(fig2)
    
    with col[2]:
        # Les acteurs les plus présents
        st.markdown('#### Acteurs les plus présents (Top 10)')
        top_acteurs = (
            filtered_df['actors_name']
            .explode()
            .value_counts()
            .head(10)
        )
        fig3, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(y=top_acteurs.index, x=top_acteurs.values, order=top_acteurs.index, palette='crest', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('')
        st.pyplot(fig3)

        # Les films les plus populaires
        st.markdown('#### Films les plus populaires (Top 10)')
        movie_by_popularity = movie_1927.groupby('originalTitle')['popularity'].max().nlargest(10)
        fig4, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(y=movie_by_popularity.index, x=movie_by_popularity.values, order=movie_by_popularity.index, palette='crest', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('')
        st.pyplot(fig4)

        # Le top 5 des pays de production les plus représentés
        st.markdown('#### Top 5 des producteurs de films')
        top_prodcountries = (movie_1927['prod_country_list'].explode().value_counts().nlargest(5))
        top_prodcountries = top_prodcountries.sort_values(ascending=False)
        fig5, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_prodcountries.values, y=top_prodcountries.index, order=top_prodcountries.index, palette='pastel', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('')
        st.pyplot(fig5)


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
    color: rgb(67.8, 84.7, 90.2) !important;
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
    background-color:rgb(183, 201, 211);
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
logo_path = "static/cine.png"

col3, col4 = st.columns([1, 3])
with col3:
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=100)
    except FileNotFoundError:
        st.error("Logo non trouvé. Vérifiez le chemin d'accès.")

with col4:
    st.markdown("""
    <style>
    .nav-bar {
        display: flex;
        align-items: center;
    }
    .nav-buttons {
        display: flex;
        justify-content: space-around;
        width: 100%;
    }
    .nav-link {
        text-decoration: none;
        padding: 10px 15px;
        border-radius: 5px;
        background-color: #f0f0f0;
        color: #333;
    }
    </style>
    <div class="nav-bar">
        <div class="nav-buttons"> <a href="?page=accueil" class="nav-link">Accueil</a>
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
    st.title("Bienvenue sur CineWhat Recommandation")
   
    st.markdown("""
    <div style='font-size:18px;'>
    <br> <b> Vous hésitez sur quel film regarder ce soir ?</b>
    <br> <b>CineWhat Recommandation</b>  est une plateforme conçue pour vous aider à découvrir des films
    qui correspondent à vos goûts grâce à une recommandation personnalisée basée sur les données.<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:18px;'>
    <br> <b> Pourquoi cette plateforme ?</b>
    <br>Ce projet a été conçu dans le cadre de mon parcours en Data Analysis afin de mettre en pratique mes compétences en collecte,
    traitement et analyse de données, tout en explorant l’univers du Machine Learning appliqué aux recommandations.<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:18px;'>
    <br> <b> Que trouverez-vous ici ?</b>
    <br>* <b>Application de recommandation</b> : Entrez vos préférences et laissez l'algorithme vous suggérer des films adaptés
    traitement et analyse de données, tout en explorant l’univers du Machine Learning appliqué aux recommandations.<br>
    * <b>Indicateurs d'analyse </b> : Visualisez des statistiques sur les films collectés, explorez les tendances du cinéma et comprenez les mécanismes derrière les recommandations.
                </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        st.markdown("""
            <style>
            .custom-button {
                font-weight: bold;
                border: 2px;
                padding: 10px 20px;
                text-align: center;
                cursor: pointer;
                background-color: white;
                color: rgb(67.8, 84.7, 90.2)
                box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.3);
                transition: background 0.3s
            }

            .custom-button:hover {
                box-shadow: rgba(183, 201, 211);
                background-color: #e6f0ff;
            }
            </style>

            <a href='?page=recommandation' style="text-decoration: none;">
                <div class="custom-button">
                    Je veux une recommandation de films
                </div>
            </a>
        """, unsafe_allow_html=True)
        
    with col8:
        st.markdown("""
            <style>
            .custom-button {
                font-weight: bold;
                border: 2px;
                padding: 10px 20px;
                text-align: center;
                cursor: pointer;
                background-color: white;
                color: rgb(67.8, 84.7, 90.2)
                box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.3);
                transition: background 0.3s
            }

            .custom-button:hover {
                box-shadow: rgba(183, 201, 211);
                background-color: #e6f0ff;
            }
            </style>

            <a href='?page=analyse' style="text-decoration: none;">
                <div class="custom-button">
                    Je veux voir les tendances du cinéma
                </div>
            </a>
        """, unsafe_allow_html=True)   

elif current_page == "recommandation":
    st.title("Bienvenue sur l'application de recommandation de films")
    
    df_movies = load_static_data()
    user_choice(df_movies)

    if "selected_movie" in st.session_state:
        selected_movie = st.session_state["selected_movie"]
        movie_details(df=df_movies, selected_movie=selected_movie)
    else:
        st.warning("Aucun film sélectionné pour le moment.")

    st.markdown("<br>", unsafe_allow_html=True)
    recommendation_show()

elif current_page == "analyse":
    st.title("Bienvenue, ici explorez des tendances du cinéma.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background-color:#f0f8ff; padding:10px; border-radius:10px; border:1px solid #cce;">
            ℹ️ <b>Info :</b> Le tableau de bord ci-dessous concerne uniquement certains films sortis entre <b>1900 et 2024</b>.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    show_kpis(df=df_movies)
else:
    st.write("Page non trouvée")