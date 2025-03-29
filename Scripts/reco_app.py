import streamlit as st
import pandas as pd
import numpy as np
import requests
#from google_images_search import GoogleImagesSearch
import ast
import plotly.express as px

# Chargement des données
df_movies4 = pd.read_csv("df_movies4.csv")
df_movies = pd.read_csv("df_movies.csv")
#movies6 = pd.read_csv("movie_new.csv") # utilisé pour les visualisations

# Initialisation de la session.  vérifier si les clés sub_page et selected_movie sont absentes dans les données utilisateurs (st.session_tate)
if "sub_page" not in st.session_state:
    st.session_state["sub_page"] = "recommandations"  # Par défaut, nous initialisons la clé "sub_page", si elle n'existe pas, comme étant la page de recommandations
if "selected_movie" not in st.session_state:
    st.session_state["selected_movie"] = None # Par défaut, nous initialisons la clé "selected_movie", si elle n'existe pas, comme étant None

# Page de l'interface de l'utilisateur
st.sidebar.title("Navigation") # titre de la barre latérale
menu = st.sidebar.radio("", ["Home", "Application"]) # définition des menus de la barre latérale

# fonction pour récupérer le poster des films
def get_image(selected_movie):
    racine = 'https://image.tmdb.org/t/p/w600_and_h900_bestv2'
    try:
        poster_url = racine + df_movies4[df_movies4['originalTitle'] == selected_movie]['poster_path'].iloc[0]
        return poster_url
    except IndexError:
        print(f"Image non trouvée pour le film {selected_movie}")
        return None

# fonction pour afficher la page d'acceuil "Home"
def display_home_page():
    if menu == "Home": 
        st.title("Bienvenue sur CineMAP")
        st.subheader('Sytème de recommandation de films : groupe 2')
        image_path = "https://www.shutterstock.com/image-vector/movie-time-600nw-324455369.jpg"
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                <img src="{image_path}" alt="Image" style="width:100%;">
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.subheader("Equipe : Aurélie, Matthieu, Philippe")

# fonction pour afficher la page des kpis (graphe dans streamlit)
# def display_kpis_page():
#     if menu == "Some KPIs":
#     # Evolution du nombre de films par genres
#         films = movies6[(movies6['titleType'] == 'movie') | (movies6['titleType'] == 'short')]
#         films['genres_liste'] = films['genres_liste'].str.split(", ")
#         films = films.explode("genres_liste").reset_index(drop=True)
#         def nettoyage_genre(genre): # après split des genres il y a des résidus de crochet et virgule, cette fonction permet de les nettoyer
#             genre_nettoye = ""
#             for var in genre:
#                 if var.isalpha():
#                     genre_nettoye += var
#             return genre_nettoye
#         films['genres_liste'] = films['genres_liste'].apply(nettoyage_genre)
#         yearly_genre_counts = films.groupby(['startYear', 'genres_liste']).size().reset_index(name='Count')
#         yearly_genre_counts.columns = ['Year', 'Genre', 'Count']
#         st.title("Evolution du nombre de films par an et par genres")
#         fig = px.bar(
#             yearly_genre_counts,
#             x="Year",
#             y="Count",
#             color="Genre",
#             barmode='stack',
#             color_discrete_sequence=px.colors.qualitative.T10  # permet de charger les palettes plotly express, directement modifiable dans le code
#         )
#         fig.update_layout(
#             xaxis_title="Année",
#             yaxis_title="Nombre de films")
#         fig.show()
#         st.plotly_chart(fig, use_container_width=True)

#         # Notes moyennes par genre
#         genre_moyenne = films.groupby('genres_liste')['averageRating'].mean().reset_index()
#         genre_moyenne.columns = ['Genre', 'Mean']
#         st.title("Notes moyennes par genres")
#         fig2 = px.histogram(
#             genre_moyenne,
#             x="Genre",
#             y="Mean",
#             color="Genre",
#             color_discrete_sequence=px.colors.qualitative.T10  # Choisir une palette de couleurs
#             )
#         fig2.update_layout(
#             xaxis_title="Genre",
#             yaxis_title="Note moyenne")
#         fig2.show()
#         st.plotly_chart(fig2, use_container_width=True)

#         # Evolution des revenus par an
#         yearly_genre_counts = films.groupby(['startYear', 'genres_liste']).size().reset_index(name='Count')
#         yearly_genre_counts.columns = ['Year', 'Genre', 'Count']
#         st.title("Evolution du nombre de films par an et par genres")
#         fig = px.bar(
#             yearly_genre_counts,
#             x="Year",
#             y="Count",
#             color="Genre",
#             barmode='stack',
#             color_discrete_sequence=px.colors.qualitative.T10  # Choisir une palette de couleurs
#         )
#         fig.update_layout(
#             xaxis_title="Année",
#             yaxis_title="Nombre de films")
#         fig.show()
#         st.plotly_chart(fig, use_container_width=True)

#         # Evolution des nos moyennes de films par an
#         films_moyenne = movies6.groupby('startYear')['averageRating'].mean().reset_index()
#         films_moyenne.columns = ['Year', 'Mean']
#         st.title("Evolution des notes moyennes des films par an")
#         fig3 = px.bar(
#             movies6,
#             x="Year",
#             y="Mean"
#             )
#         fig3.update_layout(
#             xaxis_title="Année",
#             yaxis_title="Notes moyennes")
#         fig3.show()
#         st.plotly_chart(fig3, use_container_width=True)

# permet d'afficher la page de l'application avec les recommandations
def display_application():
    if menu == "Application":
        image_path = "https://www.shutterstock.com/image-vector/movie-time-600nw-324455369.jpg"
        st.markdown(
            f"""
                <div style="display: flex; justify-content: center;">
                    <img src="{image_path}" alt="Image" style="width:100%;">
                </div>
                """, 
                unsafe_allow_html=True
            )

    # Création d'une colonne en minuscule pour faciliter les recherches
        df_movies4['originalTitle_lower'] = df_movies4['originalTitle'].str.lower()

    # Champ de saisie de l'utilisateur
        user_input = st.text_input("Saisissez un titre de film, puis tapez entrer :").lower()

    # Filtrer les titres de films en fonction de la saisie utilisateur
        if user_input:
            filtered_movies = df_movies4[df_movies4['originalTitle_lower'].str.contains(user_input, na=False)]
            movie_list = filtered_movies['originalTitle'].tolist()
        else:
            movie_list = []  # Valeur par défaut si aucun input

    # Selectbox pour afficher les suggestions
        selected_movie = st.selectbox("Choisissez un film :", movie_list)
        if selected_movie and selected_movie != "Aucun film trouvé":
            st.success(f"**{selected_movie}**")
 
    # définir la table pour récupérer les informations du film
            movie_info = df_movies4[df_movies4['originalTitle'] == selected_movie]
            if not movie_info.empty:
    # etraire et afficher des informations du film
                overview = movie_info['overview'].iloc[0]  # Résumé du film
                genres_str = ', '.join(ast.literal_eval(movie_info['genres_liste'].iloc[0]))  # tranforme la colonne genre en vraie liste de chaine de caractères, et récupère la 1st liste
                                        # retire les crochet de la liste pour avoir une chaine de caractères séparé par une virgule
    # convertir les colonnes acteurs et ordering en listes réelles
                acteurs = ast.literal_eval(movie_info['actors_name'].iloc[0])  # Liste des acteurs
                ordering = ast.literal_eval(movie_info['rang_acteur_film'].iloc[0])  # Liste des rangs
    # extraire les acteurs principaux et secondaires
                actors_with_rank = list(zip(acteurs, ordering))  # Combine acteurs et ordering
                actors_with_rank.sort(key=lambda x: x[1])  # Trie par ordre croissant de rang
    # sélectionner les 3 premiers acteurs (dont l'acteur principal)
                main_actors = []
                for actor, rank in actors_with_rank:
                    if rank == 1:
                        main_actors.append(actor)
                        break # arrêter la boucle après avoir trouvé le premier acteur
                other_actors = []
                for actor, rank in actors_with_rank:
                    if rank > 1:
                        other_actors.append(actor)
                        if len(other_actors) == 2:
                            break # arrêter la boucle après avoir récupéré 2 acteur
                act_princip = main_actors[0]
    # extraire et sélectionner le réalisateur
                realisateur_str = ', '.join(ast.literal_eval(movie_info['directors_name'].iloc[0]))  # tranformer la colonne en liste réelle et récupérer le 1er élément de la liste
    # extraire et sélectionner le scenariste
                scenariste_str = ', '.join(ast.literal_eval(movie_info['writers_name'].iloc[0]))
    # extraire et sélectionner la note
                note = movie_info['averageRating'].iloc[0]  
    # extraire et sélectionner l'année
                movie_info['startYear'] = (pd.to_datetime(movie_info['startYear'])).dt.year
                year = movie_info['startYear'].iloc[0]
    # extraire et sélectionner la durée
                time = int(movie_info['runtimeMinutes'].iloc[0])
                poster_url = get_image(selected_movie)
                st.markdown(# afficher le poster et les informations du film
                    f"""
                    <div style="display: flex; align-items: space-between;">
                                <!-- Image du film -->
                                <img src="{poster_url}"  style="margin-right: 5px;"  witdh = "180"height = "420">
                                <!-- Image du film -->
                                    <div style="max-width: 800px;">
                                        <p style="margin: 0;"><strong> Synopsis :</strong> <em> {overview} </em></p>
                                        <p style="margin: 0;"><strong> Genres :</strong> {genres_str}</p>
                                        <p style="margin: 0;"><strong> Acteur principal :</strong> {', '.join(main_actors)}</p>
                                        <p style="margin: 0;"><strong> Autres acteurs :</strong> {', '.join(other_actors)}</p>
                                        <p style="margin: 0;"><strong> Réalisé par :</strong> {realisateur_str}</p>
                                        <p style="margin: 0;"><strong> Ecrit par :</strong> {scenariste_str}</p>
                                        <p style="margin: 0;"><strong>{year} ({time} minutes)</strong></p>
                                        <p style="margin: 0;"><font size="16">{note}</font></p>
                                    </div>
                    </div>
                    """, 
                            unsafe_allow_html=True)
            if movie_info.empty:
                st.warning("Aucune information trouvée pour ce film.")
                
    try:
    # fonction similarités et choix de films
        def choix_film(titre):
            liste = df_movies[df_movies["originalTitle"]== titre]["genres_liste"].iloc[0]
            df_movies_genre = df_movies[df_movies['genres_liste'].apply(lambda x: any(genre in x for genre in liste))].copy()

            tous_les_genres = set()
            for genres in df_movies_genre['genres_liste']:
                tous_les_genres.update(genres)
            for genres in tous_les_genres:
                df_movies_genre[f'genre_{genres}'] = df_movies_genre['genres_liste'].apply(lambda x: genres in x)

            df_movies_genre = df_movies_genre.drop(['genres_liste'], axis = 1)
            df_movies_bool = df_movies_genre.select_dtypes(include='bool')
            df_movies_X = df_movies_genre.drop(['originalTitle'], axis = 1)

            def encodage_X(X, type='standard'):
                index = X.index
                X_num = X.select_dtypes('number')
                X_cat = X.select_dtypes(['object', 'category', 'string'])
            

                if type == 'standard':
                    from sklearn.preprocessing import StandardScaler
                    SN = StandardScaler()
                    X_num_SN = pd.DataFrame(SN.fit_transform(X_num), columns=X_num.columns, index=index)

                else:
                    from sklearn.preprocessing import MinMaxScaler
                    SN = MinMaxScaler()
                    X_num_SN = pd.DataFrame(SN.fit_transform(X_num), columns=X_num.columns, index=index)

                X_cat_dummies = pd.get_dummies(X_cat)
                X_encoded = pd.concat([X_num_SN, X_cat_dummies], axis=1)

                return X_encoded, SN

            X_encoded, SN = encodage_X(df_movies_X, type = "normal")
            X_encoded = pd.concat([X_encoded, df_movies_bool], axis=1)

            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=6, metric='euclidean')
            nn.fit(X_encoded)
            
            def encodage_predict(df_a_predire):
                X_num = df_a_predire.select_dtypes('number')
                X_cat = df_a_predire.select_dtypes(['object', 'category', 'string'])

                X_num_SN = pd.DataFrame(SN.transform(X_num), columns=X_num.columns).reset_index(drop=True)

                X_cat_dummies = pd.get_dummies(X_cat).reset_index(drop=True)
                X_encoded_predire = pd.concat([X_num_SN, X_cat_dummies], axis=1)

                df_predict = X_encoded_predire

                # DataFrame vide qui a les mêmes colonnes que X_encoded
                df_final = pd.DataFrame(columns=X_encoded.columns)

                # On veut que le DataFrame ait le même nombre de lignes que df_predict
                df_final = df_final.reindex(index=df_predict.index)
                # On met tous les NaN à False
                df_final = df_final.fillna(False)

                # On parcourt chaque colonne de df_predict
                # Si la colonne est présente dans X_encoded alors on la garde
                # Sinon, on la met à False
                for column in df_predict.columns:
                    if column in X_encoded.columns:
                        df_final[column] = df_predict[column]

                return df_final

            df_movies_film = df_movies_genre[df_movies_genre["originalTitle"]==titre]
            df_movies_film  = df_movies_film .drop(columns = "originalTitle")
            df_movies_bool_film = df_movies_film.select_dtypes(include='bool')
            df_movies_bool_film = df_movies_bool_film.reset_index(drop=True)
            film_encodage = encodage_predict(df_movies_film )
            colonnes = [col for col in film_encodage.columns if col.startswith('genre_')]
            film_encodage = film_encodage.drop(columns = colonnes)
            film_encodage = film_encodage.reset_index(drop=True)
            film_encodage =pd.concat([film_encodage, df_movies_bool_film], axis=1)
            distances, indices = nn.kneighbors(film_encodage)
            return df_movies_genre.iloc[indices[0,1:]]


        # initialiser des variables pour afficher les caratéristiques et affciher des films similaires

        film = choix_film(selected_movie)

        dico_info_film = {}
        liste_over = []
        liste_genre =[]
        liste_main_a = []
        liste_main_o = []
        liste_real = []
        liste_scen = []
        liste_p = []
        liste_t = []
        liste_n =[]
        liste_y = []
        liste_time = []


        # boucle pour retrouver les caractéristisques et poster des films similaires (création de liste par caractéristiques)
        for i in range(len(film)) :
            movie = film['originalTitle'].iloc[i]
            movie_info = df_movies4[df_movies4['originalTitle'] == movie]
            liste_t.append(movie)
            # Extraction des informations
            overview = movie_info['overview'].iloc[0]  # Résumé
            liste_over.append(overview)

            note = movie_info['averageRating'].iloc[0]  # Résumé
            liste_n.append(note)

            genres = ast.literal_eval(movie_info['genres_liste'].iloc[0])  # Genres
            genres_str = ', '.join(genres)  # Genres
            liste_genre.append(genres_str)

            # Conversion des colonnes acteurs et ordering en listes réelles
            acteurs = ast.literal_eval(movie_info['actors_name'].iloc[0])  # Liste des acteurs
            ordering = ast.literal_eval(movie_info['rang_acteur_film'].iloc[0])  # Liste des rangs
            # Extraction des acteurs principaux et secondaires
            actors_with_rank = list(zip(acteurs, ordering))  # Combine acteurs et ordering
            actors_with_rank.sort(key=lambda x: x[1])  # Trie par ordre croissant de rang
            # Sélectionner les 3 premiers acteurs (dont l'acteur principal)
            main_actors = [actor for actor, rank in actors_with_rank if rank == 1][:1]# acteurs avec rang 1 et >1
            liste_main_a.append( main_actors)

            other_actors = [actor for actor, rank in actors_with_rank if rank > 1][:2]# acteurs avec rang >1
            # top_actors = main_actors + other_actors
            liste_main_o.append(other_actors)

            # Extraire et sélectionner le réalisateur
            realisateur = ast.literal_eval(movie_info['directors_name'].iloc[0])  # Réalisateurs
            realisateur_str = ', '.join(realisateur)
            liste_real.append(realisateur_str)

            # Extraire et sélectionner le scenariste
            scenariste = ast.literal_eval(movie_info['writers_name'].iloc[0])  # Réalisateurs
            scenariste_str = ', '.join(scenariste)
            liste_scen.append(scenariste_str)

            #extraire l'affiche du film
            poster_url = get_image(movie)
            liste_p.append(poster_url)

            # Extraire et sélectionner l'année
            movie_info['startYear'] = (pd.to_datetime(movie_info['startYear'])).dt.year
            year = movie_info['startYear'].iloc[0]
            liste_y.append(year)

            # Extraire et sélectionner la durée
            time = int(movie_info['runtimeMinutes'].iloc[0])
            liste_time.append(time)

            # insertions des caractéristiques des films similaires dans un dictionnaire
            dico_info_film['overview'] = liste_over
            dico_info_film['genres'] = liste_genre
            dico_info_film['main_actors'] = liste_main_a
            dico_info_film['other_actors'] = liste_main_o
            dico_info_film['poster'] = liste_p
            dico_info_film['scenariste'] = liste_scen
            dico_info_film['realisateur'] = liste_real
            dico_info_film['title'] = liste_t
            dico_info_film['averageRating'] = liste_n
            dico_info_film['startYear'] = liste_y
            dico_info_film['runtimeMinutes'] = liste_time


        # affichage des poster des films similaires 
        if st.session_state["sub_page"] == "recommandations": # affichage de la sous-page de recommandation avec uniquement les affiches et les titres clicables
            st.subheader("Films dans le même genre:")
            columns = st.columns(len(film))# créer une liste d'objets sous forme de grille qui contiendra le nombre films recommandés à afficher

            for i, col in enumerate(columns):
                    with col: # "with" permet d'indiquer que les informations ci-dessous soient intégrer dans chaque colonne
                        st.image(dico_info_film["poster"][i], use_column_width=True)
                        if st.button(f"{dico_info_film['title'][i]}"): # crée un bouton à cliquer pour chaque titre
                            st.session_state["selected_movie"] = i 
                            st.session_state["sub_page"] = "details"

        
        if st.session_state["sub_page"] == "details": # initialisation d'une nouvelle sous-page affichant les détails du films recommandés sélectionné par l'utilisateur
                i = st.session_state["selected_movie"]
                st.title(f"Détails du film")
                st.image(dico_info_film["poster"][i], use_column_width=True)
                st.write(f"**Résumé :** {dico_info_film['overview'][i]}")
                st.write(f"**Genres :** {dico_info_film['genres'][i]}")
                st.write(f"**Acteur principal :** {', '.join(dico_info_film['main_actors'][i])}")
                st.write(f"**Autres acteurs :** {', '.join(dico_info_film['other_actors'][i])}")
                st.write(f"**Réalisateur :** {dico_info_film['realisateur'][i]}")
                st.write(f"**Scénariste :** {dico_info_film['scenariste'][i]}")
                st.write(f"**Année :** {dico_info_film['startYear'][i]}")
                st.write(f"**Durée :** {dico_info_film['runtimeMinutes'][i]} minutes")
                st.write(f"**Note moyenne :** {dico_info_film['averageRating'][i]}")
                
                st.button("⬅️ Retour aux recommandations") # boutton de retour qui permet de revenir à la sous_page de recommandation(liste des films recommandés)
                st.session_state["sub_page"] = "recommandations"


    

        st.subheader(f"Films avec {act_princip}:")
        

        df_acteur = df_movies4[df_movies4.apply(lambda row: any(rank == 1 and actor == act_princip   
        for actor, rank in sorted(zip(ast.literal_eval(row['actors_name']), ast.literal_eval(row['rang_acteur_film'])), key=lambda x: x[1])), axis=1)]
#On converti les chaines en listes, on associe les acteurs et leurs rangs, on les trie, on vérifie que l'acteur rang 1 correspond afin de conserver la ligne

        df_acteur = df_acteur[df_acteur['originalTitle'] != selected_movie] #On exclut le film de base
        df_acteur = df_acteur.sort_values(by='budget', ascending=False).head(5) #On garde les 5 plus gros budget 

        liste_p1 = []
        liste_t1 = []
        for i in range(len(df_acteur)) :
            movie1 = df_acteur['originalTitle'].iloc[i]
            liste_t1.append(movie1)
            poster_url1 = get_image(movie1)
            liste_p1.append(poster_url1)
            dico_info_film1 = {}
            dico_info_film1['poster'] = liste_p1
            dico_info_film1['title'] = liste_t1
        columns1 = st.columns(len(df_acteur))
        for i, col in enumerate(columns1):
                    with col: # "with" permet d'indiquer que les informations ci-dessous soient intégrer dans chaque colonne
                        st.image(dico_info_film1["poster"][i], use_column_width=True)
                        st.write(f"{dico_info_film1['title'][i]}")    
     
        
    except:
        print("")
        


# définir la navigation entre les diférenctes pages à partir des fonctions définies
if menu == "Home":
    display_home_page()
elif menu == "Application":
    display_application()
