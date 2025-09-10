import gradio as gr
import pickle
import requests

# Load data
movies = pickle.load(open('movie-recommender-system/movie_dict.pkl', 'rb'))
similarity = pickle.load(open('movie-recommender-system/similarity.pkl', 'rb'))

# OMDb API key
OMDB_API_KEY = "325fade6"

def fetch_poster_omdb(movie_title):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        poster_url = data.get('Poster', '')
        print(f"Movie: {movie_title}, Poster: {poster_url}")  # Debug print
        return poster_url if poster_url != "N/A" else ""
    except Exception as e:
        print(f"Error fetching poster for {movie_title}: {e}")
        return ""

def recommend(movie):
    # Find the index of the selected movie
    movie_index = next((idx for idx, m in enumerate(movies) if m['original_title'] == movie), None)
    if movie_index is None:
        return [], []
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    recommended_posters = []
    for i in movie_list:
        movie_title = movies[i[0]]['original_title']
        recommended_movies.append(movie_title)
        recommended_posters.append(fetch_poster_omdb(movie_title))
    return recommended_movies, recommended_posters

def gradio_recommender(selected_movie):
    names, posters = recommend(selected_movie)
    # Return list of tuples for Gallery: (image_url, caption)
    return [(poster, name) for name, poster in zip(names, posters) if poster]

movie_titles = [m['original_title'] for m in movies]

iface = gr.Interface(
    fn=gradio_recommender,
    inputs=gr.Dropdown(choices=movie_titles, label="Select a movie"),
    outputs=gr.Gallery(label="Recommended Movies", show_label=True, elem_id="gallery", columns=5, rows=1, object_fit="contain", height="auto"),
    title="Movie Recommender System"
)

iface.launch()
