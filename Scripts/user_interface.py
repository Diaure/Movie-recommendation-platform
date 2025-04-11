import streamlit as st
import pandas as pd
import joblib
import os
import subprocess
import ast
from streamlit_modal import Modal
from PIL import Image
from datetime import datetime
from scraping import fetch_movies_with_credits, fetch_upcoming_movies_with_credits
from scipy.sparse import hstack
import pickle


#***********************************************************************************
# configuration de la page
st.set_page_config(page_title="Recommandation de Films", page_icon="🎬", layout="wide")

# Image de fond (cinéma)
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
                        url("https://images.unsplash.com/photo-1542204165-65bf26472b9b?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OHx8ZmlsbXxlbnwwfHwwfHx8MA%3D%3D");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
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
        poster_url = racine + df_source[df_source['originalTitle'] == selected_movie]['poster_path'].iloc[0]
        return poster_url
    except IndexError:
        print(f"Image non trouvée pour le film {selected_movie}")
        return "logo.PNG"

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
        movie_info['startYear'] = (pd.to_datetime(movie_info['startYear'])).dt.year
        year = movie_info['startYear'].iloc[0]

        # extraire et sélectionner la durée
        time = int(movie_info['runtimeMinutes'].iloc[0])

        poster_url = get_image(selected_movie, df_movies)
        st.markdown(f""" 
        <div style="display: flex; align-items: space-between;">
                <!-- Image du film -->
                <img src="{poster_url}" style="margin-right: 10px; width:520px; height:400px;">
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


# fonction de recommandation
def recommander_films(film_titre, df, knn, scaler, tfidf_vectorizer, n_recommendations=5):

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

# fonction de recommandation permettant d'entrainer les modèles sur les nouvelles bases
def recommander_depuis_autre_df(film_titre, df_source, df_target, knn, scaler, tfidf_vectorizer, features, n_recommendations=10):
    try:
        match = df_source[df_source['originalTitle'] == film_titre]
        if match.empty:
            return f"Film '{film_titre}' non trouvé dans df_source"
        
        print("Film sélectionné :", film_titre)
        print("Exemples dans df_source :", df_source['originalTitle'].head(10).tolist())
        print("Match exact :", df_source[df_source['originalTitle'] == film_titre])

        index = match.index[0]

        # Features numériques
        X_features = df_source.loc[[index]]
        X_features = align_features(X_features, features)
        X_scaled = pd.DataFrame(scaler.transform(X_features), columns=features)
        # X_num = scaler.transform(df_source.loc[[index], features].reindex(columns=features))

        missing_features_source = [feat for feat in features if feat not in df_source.columns]
        missing_features_target = [feat for feat in features if feat not in df_target.columns]

        print("Features manquants dans df_source :", missing_features_source)
        print("Features manquants dans df_target :", missing_features_target)

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

# définir les features utilisées
# features = ['runtimeMinutes', 'averageRating', 'numVotes', 'popularity', 'budget']
# features += [col for col in df_movies.columns if col.startswith("genre_")]

# TESTS
film_test = "Titanic"  # par exemple, ou n’importe quel film que tu sais présent dans df_movies

reco_now = recommander_depuis_autre_df(
    film_titre=film_test,
    df_source=df_movies,
    df_target=df_now_playing,
    knn=knn,
    scaler=scaler,
    tfidf_vectorizer=tfidf_vectorizer,
    features=features,
    n_recommendations=10
)
st.write(reco_now)

reco_up = recommander_depuis_autre_df(
    film_titre=film_test,
    df_source=df_movies,
    df_target=df_upcoming,
    knn=knn,
    scaler=scaler,
    tfidf_vectorizer=tfidf_vectorizer,
    features=features,
    n_recommendations=10
)
st.write(reco_up)

st.write(df_now_playing[df_now_playing["originalTitle"].str.contains("White", case=False, na=False)])


# fonction pour la recommandation par acteur
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

# fonction d'affichage des recommandations
def recommendation_show():
    selected_movie = st.session_state.get("selected_movie", None)

    if not selected_movie:
        st.warning("Veuillez sélectionner un film d'abord.")
        return

    # films dans le même genre
    st.subheader(" 🎬 Films dans le même genre")
    movie_reco = recommander_films(film_titre=selected_movie, df=df_movies, knn=knn, scaler=scaler, tfidf_vectorizer=tfidf_vectorizer, n_recommendations=5)

    if isinstance(movie_reco, str):
        st.error(movie_reco)
        return

    cols = st.columns(len(movie_reco))
    for i, col in enumerate(cols):
        with col:
            st.image(get_image(movie_reco["originalTitle"].iloc[i], df_movies), use_container_width=True)
            st.caption(movie_reco["originalTitle"].iloc[i])

    # films avec l'acteur principal
    act_princip = st.session_state.get("main_actor", None)
    if act_princip:
        st.subheader(f"Films avec {act_princip}")
        movie_main_actor = recommander_par_acteur(act_princip, df_movies)

        cols_actor = st.columns(len(movie_main_actor))
        for i, col in enumerate(cols_actor):
            with col:
                st.image(get_image(movie_main_actor["originalTitle"].iloc[i], df_movies), use_container_width=True)
                st.caption(movie_main_actor["originalTitle"].iloc[i])
    else:
        st.info("Acteur principal non disponible.")

    # films actuellement au cinéma
    st.subheader("Films actuellement au cinéma")
    movie_now_playing = recommander_depuis_autre_df(film_titre=selected_movie, df_source=df_movies, df_target=df_now_playing, knn=knn, scaler=scaler, tfidf_vectorizer=tfidf_vectorizer, features=features, n_recommendations=5)
    print(len(movie_now_playing))


    if isinstance(movie_now_playing, pd.DataFrame) and not movie_now_playing.empty:
        num_movies = movie_now_playing.shape[0]
        print(type(movie_now_playing))
        print(movie_now_playing)
        print(movie_now_playing.shape)
        cols_now = st.columns(num_movies)
        for i, col in enumerate(cols_now):
            with col:
                st.image(get_image(movie_now_playing["originalTitle"].iloc[i], df_now_playing), use_container_width=True)
                st.caption(movie_now_playing["originalTitle"].iloc[i])
    else:
        st.info("Aucun film à recommander pour les séances en cours.")

    # films bientôt au cinéma
    st.subheader("Films prochainement au cinéma")
    movie_upcoming = recommander_depuis_autre_df(film_titre=selected_movie, df_source=df_movies, df_target=df_upcoming, knn=knn, scaler=scaler, tfidf_vectorizer=tfidf_vectorizer, features=features, n_recommendations=5)
    
    if isinstance(movie_upcoming, str):
        st.error(movie_upcoming)
    else:
        cols_now = st.columns(len(movie_upcoming))
        for i, col in enumerate(cols_now):
            with col:
                st.image(get_image(movie_upcoming["originalTitle"].iloc[i], df_upcoming), use_column_width=True)
                st.caption(movie_upcoming["originalTitle"].iloc[i])


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
logo_path = "static/cine.png"
# st.markdown("""
# <div class="nav-bar">
#     <div class="logo-h1">
#         <img src="static/Capture.PNG" alt="Votre Logo" style="height: 100px; margin-right: 30px;">
#     </div>
#     <div class="nav-buttons">  <a href="?page=accueil" class="nav-link">Accueil</a>
#         <a href="?page=recommandation" class="nav-link">Application</a>
#             <a href="?page=analyse" class="nav-link">Indicateurs clés films</a>
#     </div>
# </div>
# """, unsafe_allow_html=True)

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
    col5, col6 = st.columns([1, 2])
    with col5:
        st.image(logo_path, use_container_width=True)
    
    with col6:
        st.subheader("Vous hésitez sur quel film regarder ce soir ?")
        st.markdown("""
        <div style='font-size:18px;'>
        <br> <b>CineWhat Recommandation</b>  est une plateforme conçue pour vous aider à découvrir des films
        qui correspondent à vos goûts grâce à une recommandation personnalisée basée sur les données.<br>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Pourquoi cette plateforme ?")
        st.markdown("""
        <div style='font-size:18px;'>
        <br>Ce projet a été conçu dans le cadre de mon parcours en Data Analysis afin de mettre en pratique mes compétences en collecte,
        traitement et analyse de données, tout en explorant l’univers du Machine Learning appliqué aux recommandations.<br>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Que trouverez-vous ici ?")
        st.markdown("""
        <div style='font-size:18px;'>
        <br>* <b>Application de recommandation</b> : Entrez vos préférences et laissez l'algorithme vous suggérer des films adaptés
        traitement et analyse de données, tout en explorant l’univers du Machine Learning appliqué aux recommandations.<br>
        * <b>Indicateurs d'analyse </b> : Visualisez des statistiques sur les films collectés, explorez les tendances du cinéma et comprenez les mécanismes derrière les recommandations.
                    </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
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
            Je veux une recommandation de films
        </div>
        """, unsafe_allow_html=True)
        
    with col8:
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
    # st.markdown("""
    # <img src="static/Capture.JPG" style="float: left" alt="Image" style="width:100%;"/> </div>""", unsafe_allow_html=True)
    
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
    st.title("Bienvenue, ici explorez les tendances du cinéma.")
else:
    st.write("Page non trouvée")