# 🎬 Movie Platform - CineWhat Recommandation


Bienvenue sur la plateforme de recommandation de films ! Cette application interactive, développée avec Streamlit, vous permet de découvrir des recommandations personnalisées basées sur vos goûts cinématographiques.

## 🚀 Objectif du projet

L’objectif de cette plateforme est de proposer une expérience utilisateur intuitive pour explorer et découvrir des films selon différents critères, tout en exploitant des techniques de traitement de données et de scraping web.

## Navigation

### 🏠 Accueil
Une page d’introduction qui présente la plateforme, son objectif et les différentes fonctionnalités accessibles.

### 🎥 Application
C’est ici que tout se passe ! L’utilisateur peut :

* Choisir un film dans la liste proposée.

* Recevoir des recommandations basées sur :

    * Le genre du film sélectionné.

    * L’acteur principal du film.

    * Les films actuellement au cinéma.

    * Les films à venir, obtenus via scraping depuis l’API de TMDB

Une fonctionnalité de **`mise à jour forcée`** permet de rafraîchir manuellement les données des films actuellement diffusés ou à venir, pour s'assurer que les recommandations soient toujours pertinentes.

### 📊 Tendances

Un tableau de bord permettant d'explorer :

* Les genres les plus populaires

* Les acteurs les plus représentés

* Les pays de production dominants

* Les films les plus populaires

* Et d'autres KPIs clés sur l'univers cinématographique

## 🛠️ Technologies utilisées
* Python (Pandas) pour la manipulation des données

* Streamlit pour l’interface utilisateur

* Seaborn & Plotly pour les visualisations

* Requests & BeautifulSoup pour le web scraping.

* TMDB API pour récupérer les informations des films.

## Fonctionnalités clés
* Recommandation intelligente de films à partir d’un film sélectionné.

* Analyse des genres et acteurs les plus présents.

* Données actualisées en temps réel grâce à l’intégration API.

* Interface simple et accessible, utilisable sans connaissance technique.
