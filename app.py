import streamlit as st
import pickle
import pandas as pd
import requests

# Function to fetch movie poster URL from TMDB API
def fetch_poster(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'
    response = requests.get(url)
    data = response.json()
    if 'poster_path' in data:
        return "https://image.tmdb.org/t/p/w500" + data['poster_path']
    else:
        return None

# Function to recommend similar movies
def recommend(movie):
    movie_index = movies[movies['original_title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []
    
    for i in movies_list:
        movie_id = movies.iloc[i[0]].id_x
        recommended_movies.append(movies.iloc[i[0]].original_title)
        poster_url = fetch_poster(movie_id)
        recommended_posters.append(poster_url)
    
    return recommended_movies, recommended_posters

# Load movie data and similarity matrix
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Streamlit interface

st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Select a movie',
    movies['original_title'].values)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    col1,col2,col3,col4,col5=st.columns(5)

    with col1:
        st.header(names[0])
        st.image(posters[0])

    with col2:
        st.header(names[1])    
        st.image(posters[1])
    with col3:
        st.header(names[2]) 
        st.image(posters[2]) 
    with col4:
        st.header(names[3])  
        st.image(posters[3]) 
    with col5:
        st.header(names[4])  
        st.image(posters[4])   


