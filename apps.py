# app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# --- Load Data ---

# Load movies
movies = pd.read_csv('movies.dat', sep='::', engine='python', header=None, names=['movieId', 'title', 'genres'], encoding='ISO-8859-1')

# Load ratings
ratings = pd.read_csv('ratings.dat', sep='::', engine='python', header=None, names=['userId', 'movieId', 'rating', 'timestamp'], encoding='ISO-8859-1')

# --- Preprocessing ---

# Fill missing genres
movies['genres'] = movies['genres'].fillna('')

# TF-IDF on genres for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a reverse map of movie titles and indices
movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# --- Build Collaborative Filtering Model (SVD) ---

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

model = SVD()
model.fit(trainset)

# --- Recommendation Functions ---

# Collaborative Filtering Recommendation
def recommend_cf(user_id, n_recommendations=10):
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    all_movies = movies['movieId'].tolist()

    unrated_movies = [movie for movie in all_movies if movie not in user_rated_movies]

    predictions = []
    for movie_id in unrated_movies:
        pred = model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended_movie_ids = [movie_id for (movie_id, _) in predictions[:n_recommendations]]

    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]

    return recommended_movies[['title', 'genres']]

# Content-Based Recommendation
def recommend_content(title, n_recommendations=10):
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]

    return movies[['title', 'genres']].iloc[movie_indices]

# --- Streamlit App ---

st.title(' Hybrid Movie Recommendation System')

# Input: User ID
user_id = st.number_input('Enter your User ID:', min_value=1, step=1)

# Input: Favorite Movie
movie_list = movies['title'].values
selected_movie = st.selectbox('Select your Favorite Movie:', movie_list)

if st.button('Recommend'):
    with st.spinner('Finding best movies for you...'):
        # Collaborative Filtering Recommendations
        st.subheader(' Collaborative Filtering Recommendations')
        cf_recommendations = recommend_cf(user_id)
        st.dataframe(cf_recommendations)

        # Content-Based Recommendations
        st.subheader(' Content-Based Recommendations')
        cb_recommendations = recommend_content(selected_movie)
        st.dataframe(cb_recommendations)

        # Run streamlit run "E:\Data Science Project Arman Somai Devyani\ml-1m\apps.py"


