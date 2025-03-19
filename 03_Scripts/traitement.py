import warnings
warnings.filterwarnings("ignore") 
import joblib
from scraping import fetch_movies_with_credits, fetch_upcoming_movies_with_credits

# créer une liste à partir des chaines de caractère
def split_chaine_en_liste(x):
    if isinstance(x, str):
        return x.split(',') 
    else:
        return x
    
# période de films
def decennie(date):
    date = str(date)
    if date < '1910-01-01':
        return '1910'
    elif date < '1920-01-01':
        return '1910'
    elif date < '1930-01-01':
        return '1920'
    elif date < '1940-01-01':
        return '1930'
    elif date < '1950-01-01':
        return '1940'
    elif date < '1960-01-01':
        return '1950'
    elif date < '1970-01-01':
        return '1960'
    elif date < '1980-01-01':
        return '1970'
    elif date < '1990-01-01':
        return '1980'
    elif date < '2000-01-01':
        return '1990'
    elif date < '2010-01-01':
        return '2000'
    elif date < '2020-01-01':
        return '2010'
    else: 
        return '2020'

# fonction de traitement des données
def traiter_donnees_film(df):
    dfs = df.copy()

    # transformer la colonne "genres" en liste
    dfs["genres_liste"] = dfs["genres"].apply(split_chaine_en_liste)

    # extraire les genres uniques
    tous_les_genres = set()
    for genres in dfs["genres_liste"]:
        tous_les_genres.update(genres)

    # créer des colonnes binaires pour chaque genre unique
    for genre in tous_les_genres:
        def genre_present(x):
            return int(genre in x)
    
    dfs[f'genre_{genre}'] = dfs["genres_liste"].apply(genre_present)

    dfs["periode"] = dfs["release_date"].apply(decennie)

    # Insérer la colonne "periode" avant "budget"
    position_budget = df.columns.get_loc("budget")
    df.insert(position_budget, "periode", df.pop("periode"))

    return dfs


df_movies = joblib.load("..\BD_A_IGNORE\df_movies.pkl")

movie_data = fetch_movies_with_credits()
df_now_playing = movie_data[~movie_data["originalTitle"].isin(df_movies["originalTitle"])]
df_now_playing = df_now_playing.apply(traiter_donnees_film)

upcoming_movie_data = fetch_upcoming_movies_with_credits()
df_upcoming_movie_data = upcoming_movie_data[~upcoming_movie_data["originalTitle"].isin(df_movies["originalTitle"])]
df_upcoming_movie_data = df_upcoming_movie_data.apply(traiter_donnees_film)

joblib.dump(df_now_playing, "../BD_A_IGNORE/df_now_playing.pkl") # enregistrer la base sans utiliser de csv
joblib.dump(df_upcoming_movie_data, "../BD_A_IGNORE/df_upcoming_movie.pkl") # enregistrer la base sans utiliser de csv