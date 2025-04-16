# chargement des librairies, clé API, base url
import requests
import pandas as pd
import os
from dotenv import load_dotenv
import joblib
from datetime import datetime

load_dotenv()
api_key = os.getenv("TMDB_API_KEY")

if not api_key:
    print("Erreur : clé API non trouvée.")
    exit()

base_url = "https://api.themoviedb.org/3"


# Récupérer les correspondances des noms de genres et ids sur TMDB
def get_genre_mapping(api_key):
    load_dotenv()
    api_key = os.getenv("TMDB_API_KEY")

    if not api_key:
        print("Erreur : clé API non trouvée.")
        exit()

    url = f'{base_url}/genre/movie/list?api_key={api_key}&language=en-EN'
    response = requests.get(url)
    
    if response.status_code == 200:
        genres = response.json().get("genres", [])
        return {genre["id"]: genre["name"] for genre in genres}
    else:
        print("Erreur lors de la récupération des genres.")
        return {}

# Récupérer les films "Now Playing"(actuellement au cinéma) de TMDB
def get_now_playing_movies(pages=100): 
    movies = []
    for page in range(1, pages + 1):
        url = f"{base_url}/movie/now_playing?api_key={api_key}&language=fr-FR&page={page}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            movies.extend(data['results'])
        else:
            print(f" Erreur HTTP {response.status_code} sur la page {page}")
    return movies

# on récupère l'id du film
def get_movie_credits(movie_id):
    url = f"{base_url}/movie/{movie_id}/credits?api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# on récupère les détails du film
def get_movie_details(movie_id):
    url = f"{base_url}/movie/{movie_id}?api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# on récupère toutes les équipes puis on combine tout dans une base
def fetch_movies_with_credits():
    movies = get_now_playing_movies()
    movie_data = []
    today = datetime.today().date()

    for movie in movies:
        movie_id = movie["id"]
        
        details = get_movie_details(movie_id)
        if not details:
            print(f"Impossible de récupérer les détails pour {movie_id}")
            continue

        # Filtrer les films avec une date de sortie future ou aujourd'hui
        release_date_str = details.get("release_date")
        if not release_date_str:
            print(f"Date de sortie manquante pour {movie_id}")
            continue
        try:
            release_date = datetime.strptime(release_date_str, "%Y-%m-%d").date()
            if release_date > today:
                continue
        except Exception as e:
            print(f"Erreur sur la date pour {movie_id} : {e}")
            continue
        
        genres_mapping = get_genre_mapping(api_key=api_key)
        genre_ids = movie.get("genre_ids", [])
        genres = [genres_mapping.get(genre_id) for genre_id in genre_ids if genres_mapping.get(genre_id)]

        # récupérer les crédits du film
        credits = get_movie_credits(movie_id)
        if not credits:
            print(f"Impossible de récupérer les crédits pour le film {movie_id})")
            continue

        actors = []
        actors_rank = []
        for index, cast in enumerate(credits["cast"]):
            actors.append(cast["name"])
            actors_rank.append(index + 1)

        directors = []
        for crew in credits["crew"]:
            if crew["job"] == "Director":
                directors.append(crew["name"])

        writers = []
        for crew in credits["crew"]:
            if crew["job"] == "Writer":
                writers.append(crew["name"])

        producers = []
        for crew in credits["crew"]:
            if crew["job"] == "Producer":
                producers.append(crew["name"])

        cinematographers = []
        for crew in credits["crew"]:
            if crew["job"] in ["Cinematographer", "Director of Photography"]:
                cinematographers.append(crew["name"])

        editors = []
        for crew in credits["crew"]:
            if crew["job"] == "Editor":
                editors.append(crew["name"])

        # Ajouter les données dans la liste
        movie_data.append({
            "id": movie_id,
            "originalTitle": details["original_title"],
            "startYear": details["release_date"],
            "runtimeMinutes": details.get("runtime"),
            "budget": details.get("budget"),
            "averageRating": details["vote_average"],
            "numVotes": details["vote_count"],
            "popularity": details["popularity"],
            "overview": details["overview"],
            "poster_path": details["poster_path"],
            "genres": ", ".join(genres),
            "actors_name": ", ".join(actors),
            "actors_rank": ", ".join(map(str, actors_rank)),
            "directors_name": ", ".join(directors),
            "writers_name": ", ".join(writers),
            "producers_name": ", ".join(producers),
            "cinematographers_name": ", ".join(cinematographers),
            "editors_name": ", ".join(editors)
        })

    return pd.DataFrame(movie_data)

# Récupérer les films bientôt au cinéma de TMDB
def get_upcoming_movies(pages=100): 
    upcoming_movies = []
    for page in range(1, pages + 1):
        url = f"{base_url}/movie/upcoming?api_key={api_key}&language=fr-FR&page={page}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            upcoming_movies.extend(data['results'])
        else:
            print(f" Erreur HTTP {response.status_code} sur la page {page}")
    return upcoming_movies


def get_upcoming_movie_credits(movie_id):
    url = f"{base_url}/movie/{movie_id}/credits?api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# on récupère les détails du film
def get_upcoming_movie_details(movie_id):
    url = f"{base_url}/movie/{movie_id}?api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# on récupère tous les équipes puis on combine tout dans une base
def fetch_upcoming_movies_with_credits():
    upcoming_movies = get_upcoming_movies()
    upcoming_movie_data = []
    today = datetime.today().date()

    for movie in upcoming_movies:
        movie_id = movie["id"]
        
        upcoming_details = get_upcoming_movie_details(movie_id)
        if not upcoming_details:
            print(f"Impossible de récupérer les détails pour {movie_id}")
            continue

         # Filtrer les films avec une date de sortie future ou aujourd'hui
        release_date_str = upcoming_details.get("release_date")
        if not release_date_str:
            print(f"Date de sortie manquante pour {movie_id}")
            continue
        try:
            release_date = datetime.strptime(release_date_str, "%Y-%m-%d").date()
            if release_date <= today:
                continue
        except Exception as e:
            print(f"Erreur sur la date pour {movie_id} : {e}")
            continue
        
        genres_mapping = get_genre_mapping(api_key=api_key)
        genre_ids = movie.get("genre_ids", [])
        genres = [genres_mapping.get(genre_id) for genre_id in genre_ids if genres_mapping.get(genre_id)]

        # récupérer les crédits du film
        upcoming_credits = get_upcoming_movie_credits(movie_id)
        if not upcoming_credits:
            print(f"Impossible de récupérer les crédits pour le film {movie_id})")
            continue

        actors = []
        actors_rank = []
        for index, cast in enumerate(upcoming_credits["cast"]):
            actors.append(cast["name"])
            actors_rank.append(index + 1)

        directors = []
        for crew in upcoming_credits["crew"]:
            if crew["job"] == "Director":
                directors.append(crew["name"])

        writers = []
        for crew in upcoming_credits["crew"]:
            if crew["job"] == "Writer":
                writers.append(crew["name"])

        producers = []
        for crew in upcoming_credits["crew"]:
            if crew["job"] == "Producer":
                producers.append(crew["name"])

        cinematographers = []
        for crew in upcoming_credits["crew"]:
            if crew["job"] in ["Cinematographer", "Director of Photography"]:
                cinematographers.append(crew["name"])

        editors = []
        for crew in upcoming_credits["crew"]:
            if crew["job"] == "Editor":
                editors.append(crew["name"])

        # Ajouter les données dans la liste
        upcoming_movie_data.append({
            "id": movie_id,
            "originalTitle": upcoming_details["original_title"],
            "startYear": upcoming_details["release_date"],
            "runtimeMinutes": upcoming_details.get("runtime"),
            "budget": upcoming_details.get("budget"),
            "averageRating": upcoming_details["vote_average"],
            "numVotes": upcoming_details["vote_count"],
            "popularity": upcoming_details["popularity"],
            "overview": upcoming_details["overview"],
            "poster_path": upcoming_details["poster_path"],
            "genres": ", ".join(genres),
            "actors_name": ", ".join(actors),
            "actors_rank": ", ".join(map(str, actors_rank)),
            "directors_name": ", ".join(directors),
            "writers_name": ", ".join(writers),
            "producers_name": ", ".join(producers),
            "cinematographers_name": ", ".join(cinematographers),
            "editors_name": ", ".join(editors)
        })
        
    return pd.DataFrame(upcoming_movie_data)

if __name__ == "__main__":
    # Création du dossier s'il n'existe pas
    os.makedirs("BD_A_IGNORE", exist_ok=True)

    # Extraction des films en salle
    df_now_playing = fetch_movies_with_credits()
    df_now_playing.to_pickle("BD_A_IGNORE/df_now_playing.pkl")
    print("✅ df_now_playing.pkl sauvegardé avec succès.")

    # Extraction des films à venir
    df_upcoming = fetch_upcoming_movies_with_credits()
    df_upcoming.to_pickle("BD_A_IGNORE/df_upcoming.pkl")
    print("✅ df_upcoming.pkl sauvegardé avec succès.")