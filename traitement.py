import pandas as pd
import warnings
warnings.filterwarnings("ignore") 
import joblib
import os
from scraping import fetch_movies_with_credits, fetch_upcoming_movies_with_credits

warnings.filterwarnings("ignore")

# Construire un chemin absolu vers le dossier BD_A_IGNORE
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BD_PATH = os.path.join(CURRENT_DIR, "BD_A_IGNORE")
os.makedirs(BD_PATH, exist_ok=True)

# créer une liste à partir des chaines de caractère
def split_chaine_en_liste(x):
    if isinstance(x, str):
        return [genre.strip() for genre in x.split(',')]
    else:
        return x
    
# période de films
def decennie(date):
    date = pd.to_datetime(date, errors="coerce")
    if pd.isna(date):
        return 'inconnu'  # Gérer les valeurs manquantes
    elif date.year < 1920:
        return '1910'
    elif date.year < 1930:
        return '1920'
    elif date.year < 1940:
        return '1930'
    elif date.year < 1950:
        return '1940'
    elif date.year < 1960:
        return '1950'
    elif date.year < 1970:
        return '1960'
    elif date.year < 1980:
        return '1970'
    elif date.year < 1990:
        return '1980'
    elif date.year < 2000:
        return '1990'
    elif date.year < 2010:
        return '2000'
    elif date.year < 2020:
        return '2010'
    else: 
        return '2020'

# fonction de traitement des données
def traiter_donnees_film(df):

    if "genres" not in df.columns:
        raise ValueError("La colonne 'genres' est absente du DataFrame")
    
    dfs = df.copy()

    # transformer les colonne en "vraies" listes
    dfs["genres_liste"] = dfs["genres"].apply(split_chaine_en_liste)
    dfs["actors_name"] = dfs["actors_name"].apply(split_chaine_en_liste)
    dfs["actors_rank"] = dfs["actors_rank"].apply(split_chaine_en_liste)
    dfs["directors_name"] = dfs["directors_name"].apply(split_chaine_en_liste)
    dfs["writers_name"] = dfs["writers_name"].apply(split_chaine_en_liste)
    dfs["producers_name"] = dfs["producers_name"].apply(split_chaine_en_liste)
    dfs["cinematographers_name"] = dfs["cinematographers_name"].apply(split_chaine_en_liste)
    dfs["editors_name"] = dfs["editors_name"].apply(split_chaine_en_liste)

    # extraire les genres uniques
    tous_les_genres = set()
    for genres in dfs["genres_liste"]:
        tous_les_genres.update(genres)

    # créer des colonnes binaires pour chaque genre unique
    for genre in tous_les_genres:
        dfs[f'genre_{genre}'] = dfs["genres_liste"].apply(lambda x: int(genre in x))

    dfs["periode"] = dfs["startYear"].apply(decennie)

    return dfs


# df_movies = joblib.load("../BD_A_IGNORE/df_movies.pkl")
df_movies = joblib.load(os.path.join(BD_PATH, "df_movies.pkl"))

movie_data = fetch_movies_with_credits()
df_now_playing = movie_data[~movie_data["originalTitle"].isin(df_movies["originalTitle"])]
df_now_playing = traiter_donnees_film(df_now_playing)

upcoming_movie_data = fetch_upcoming_movies_with_credits()
df_upcoming_movie_data = upcoming_movie_data[~upcoming_movie_data["originalTitle"].isin(df_movies["originalTitle"])]
df_upcoming_movie_data = traiter_donnees_film(df_upcoming_movie_data)

# joblib.dump(df_now_playing, "../BD_A_IGNORE/df_now_playing.pkl") # enregistrer la base sans utiliser de csv
# joblib.dump(df_upcoming_movie_data, "../BD_A_IGNORE/df_upcoming_movie.pkl") # enregistrer la base sans utiliser de csv

joblib.dump(df_now_playing, os.path.join(BD_PATH, "df_now_playing.pkl"))
joblib.dump(df_upcoming_movie_data, os.path.join(BD_PATH, "df_upcoming_movie.pkl"))

print("Bases enregistrées avec succès !")
print(df_now_playing.columns)
print(df_upcoming_movie_data.columns)