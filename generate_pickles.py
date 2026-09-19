import ast
import json
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("MOVIE_DATA_DIR", BASE_DIR / "data"))
MOVIES_CSV = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = DATA_DIR / "tmdb_5000_credits.csv"
MOVIE_DICT_PATH = BASE_DIR / "movie_dict.pkl"
SIMILARITY_PATH = BASE_DIR / "similarity.pkl"


def parse_structured(value):
    if pd.isna(value) or not str(value).strip():
        return []
    text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []


def get_director(value):
    for item in parse_structured(value):
        if item.get("job") == "Director":
            return item.get("name", "")
    return ""


def get_names(value):
    return " ".join(
        item.get("name", "") for item in parse_structured(value) if item.get("name")
    )


def get_top_cast(value, limit=3):
    return " ".join(
        item.get("name", "")
        for item in parse_structured(value)[:limit]
        if item.get("name")
    )


if not MOVIES_CSV.exists() or not CREDITS_CSV.exists():
    raise FileNotFoundError(
        "Dataset files not found. Put tmdb_5000_movies.csv and "
        "tmdb_5000_credits.csv inside data/, or set MOVIE_DATA_DIR."
    )

movies_df = pd.read_csv(MOVIES_CSV)
credits_df = pd.read_csv(CREDITS_CSV).rename(columns={"movie_id": "id"})
df = movies_df.merge(credits_df, on="id")

df["director"] = df["crew"].apply(get_director)
df["top_cast"] = df["cast"].apply(get_top_cast)
df["genres"] = df["genres"].apply(get_names)
df["keywords"] = df["keywords"].apply(get_names)

for column in ["overview", "genres", "director", "top_cast", "keywords"]:
    df[column] = df[column].fillna("").astype(str)

df["tags"] = (
    df["overview"] + " " + df["genres"] + " " + df["director"] + " "
    + df["top_cast"] + " " + df["keywords"]
).str.lower()

vectorizer = CountVectorizer(max_features=5000, stop_words="english")
vectors = vectorizer.fit_transform(df["tags"]).toarray()
similarity = cosine_similarity(vectors)

movie_dict = df[
    ["id", "original_title", "genres", "director", "top_cast", "keywords", "overview"]
].to_dict("records")

with open(MOVIE_DICT_PATH, "wb") as file:
    pickle.dump(movie_dict, file)

with open(SIMILARITY_PATH, "wb") as file:
    pickle.dump(similarity, file)

print(f"Generated {MOVIE_DICT_PATH}")
print(f"Generated {SIMILARITY_PATH}")
print(f"Movies: {len(movie_dict)}")
print(f"Feature matrix: {vectors.shape}")
