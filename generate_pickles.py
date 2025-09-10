import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
df_movies = pd.read_csv('C:/Users/siraj/Desktop/New folder (5)/tmdb_5000_movies.csv')
df_credits = pd.read_csv('C:/Users/siraj/Desktop/New folder (5)/tmdb_5000_credits.csv')

# Merge on 'id' and 'movie_id'
df_credits.rename(columns={'movie_id': 'id'}, inplace=True)
df = df_movies.merge(df_credits, on='id')

# Select relevant features
def get_director(x):
    for i in eval(x):
        if i['job'] == 'Director':
            return i['name']
    return ''

def get_top_cast(x):
    return ' '.join([i['name'] for i in eval(x)[:3]])

def get_genres(x):
    return ' '.join([i['name'] for i in eval(x)])

def get_keywords(x):
    return ' '.join([i['name'] for i in eval(x)])

df['director'] = df['crew'].apply(get_director)
df['top_cast'] = df['cast'].apply(get_top_cast)
df['genres'] = df['genres'].apply(get_genres)
df['keywords'] = df['keywords'].apply(get_keywords)
df['tags'] = df['overview'] + ' ' + df['genres'] + ' ' + df['director'] + ' ' + df['top_cast'] + ' ' + df['keywords']
df['tags'] = df['tags'].str.lower()
df['tags'] = df['tags'].fillna('')

# Vectorize tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

# Compute similarity matrix
similarity = cosine_similarity(vectors)

# Prepare movie dictionary
movie_dict = df[['id', 'original_title']].to_dict('records')

# Save pickle files
try:
    with open('movie_dict.pkl', 'wb') as f:
        pickle.dump(movie_dict, f)
    print('movie_dict.pkl saved successfully.')
except Exception as e:
    print(f'Error saving movie_dict.pkl: {e}')

try:
    with open('similarity.pkl', 'wb') as f:
        pickle.dump(similarity, f)
    print('similarity.pkl saved successfully.')
except Exception as e:
    print(f'Error saving similarity.pkl: {e}')
