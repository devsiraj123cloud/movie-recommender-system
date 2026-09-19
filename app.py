import os
import pickle
import random
import sqlite3
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import requests

BASE_DIR = Path(__file__).resolve().parent
MOVIE_DICT_PATH = BASE_DIR / "movie_dict.pkl"
SIMILARITY_PATH = BASE_DIR / "similarity.pkl"
DB_PATH = BASE_DIR / "movie_recommender.db"

OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")
OMDB_URL = "https://www.omdbapi.com/"

if not MOVIE_DICT_PATH.exists() or not SIMILARITY_PATH.exists():
    raise FileNotFoundError(
        "movie_dict.pkl or similarity.pkl is missing. Run \`python generate_pickles.py\` first."
    )

with open(MOVIE_DICT_PATH, "rb") as f:
    movies = pickle.load(f)
with open(SIMILARITY_PATH, "rb") as f:
    similarity = pickle.load(f)

if not movies:
    raise ValueError("movie_dict.pkl is empty.")

movie_titles = [movie["original_title"] for movie in movies]


def normalize(value):
    return str(value).lower() if value else ""


def genres_for(movie):
    genres = movie.get("genres", "")
    if isinstance(genres, list):
        return [normalize(g.get("name", g)) for g in genres]
    return [g.strip() for g in str(genres).lower().split() if g.strip()]


class EpsilonGreedyBandit:
    """Lightweight epsilon-greedy bandit updated by user feedback."""

    def __init__(self, epsilon=0.10):
        self.epsilon = epsilon
        self.counts = {}
        self.values = {}

    def value(self, movie_id):
        return self.values.get(str(movie_id), 0.0)

    def choose(self, candidates):
        if not candidates:
            return None
        untried = [m for m in candidates if str(m.get("id")) not in self.counts]
        if untried and random.random() < 0.5:
            return random.choice(untried)
        if random.random() < self.epsilon:
            return random.choice(candidates)
        return max(candidates, key=lambda m: self.value(m.get("id")))

    def update(self, movie_id, reward):
        key = str(movie_id)
        count = self.counts.get(key, 0) + 1
        old_value = self.values.get(key, 0.0)
        self.counts[key] = count
        self.values[key] = old_value + (reward - old_value) / count


class MovieRecommender:
    def __init__(self):
        self.bandit = EpsilonGreedyBandit()
        self.poster_cache = {}
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    movie_id INTEGER,
                    movie_title TEXT,
                    action TEXT,
                    timestamp TEXT
                )
            """)

    def content_candidates(self, selected_movie, limit=20):
        if not selected_movie:
            return []
        try:
            index = movie_titles.index(selected_movie)
        except ValueError:
            return []
        distances = np.asarray(similarity[index])
        ranked = np.argsort(distances)[::-1]
        results = []
        for i in ranked:
            if i == index:
                continue
            results.append(movies[int(i)])
            if len(results) >= limit:
                break
        return results

    def contextual_candidates(self, context, limit=20):
        context_genres = {
            "general": set(),
            "evening": {"action", "thriller", "horror"},
            "family": {"family", "animation", "comedy"},
            "romantic": {"romance", "drama"},
            "action": {"action", "adventure", "thriller"},
        }
        wanted = context_genres.get(context, set())
        if not wanted:
            return movies[:limit]
        matched = [
            movie for movie in movies
            if wanted.intersection(set(genres_for(movie)))
        ]
        return matched[:limit]

    def hybrid_candidates(self, selected_movie, context, limit=20):
        content = self.content_candidates(selected_movie, limit=50)
        context_movies = self.contextual_candidates(context, limit=50)
        by_id = {}
        for movie in content + context_movies:
            by_id[movie["id"]] = movie
        if not by_id:
            return []

        try:
            selected_index = movie_titles.index(selected_movie)
            content_scores = np.asarray(similarity[selected_index])
        except ValueError:
            content_scores = None

        context_genres = {
            "general": set(),
            "evening": {"action", "thriller", "horror"},
            "family": {"family", "animation", "comedy"},
            "romantic": {"romance", "drama"},
            "action": {"action", "adventure", "thriller"},
        }
        wanted = context_genres.get(context, set())

        scored = []
        for movie in by_id.values():
            index = movie_titles.index(movie["original_title"])
            content_score = (
                float(content_scores[index]) if content_scores is not None else 0.0
            )
            context_score = 1.0 if wanted.intersection(set(genres_for(movie))) else 0.0
            bandit_score = self.bandit.value(movie.get("id"))
            score = 0.60 * content_score + 0.25 * context_score + 0.15 * bandit_score
            scored.append((score, movie))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [movie for _, movie in scored[:limit]]

    def recommend(self, selected_movie, context, algorithm):
        if algorithm == "Content-Based":
            candidates = self.content_candidates(selected_movie)
        elif algorithm == "Context-Aware":
            candidates = self.contextual_candidates(context)
        elif algorithm == "Multi-Armed Bandit":
            candidates = self.content_candidates(selected_movie, limit=30)
            chosen = []
            remaining = list(candidates)
            while remaining and len(chosen) < 5:
                movie = self.bandit.choose(remaining)
                if movie is None:
                    break
                chosen.append(movie)
                remaining.remove(movie)
            return chosen
        else:
            candidates = self.hybrid_candidates(selected_movie, context)
        return candidates[:5]

    def fetch_details(self, title):
        if title in self.poster_cache:
            return self.poster_cache[title]
        if not OMDB_API_KEY:
            details = {"poster": "", "rating": "N/A", "year": "N/A"}
            self.poster_cache[title] = details
            return details
        try:
            response = requests.get(
                OMDB_URL,
                params={"apikey": OMDB_API_KEY, "t": title, "plot": "short"},
                timeout=8,
            )
            data = response.json()
            details = {
                "poster": "" if data.get("Poster") == "N/A" else data.get("Poster", ""),
                "rating": data.get("imdbRating", "N/A"),
                "year": data.get("Year", "N/A"),
            }
        except requests.RequestException:
            details = {"poster": "", "rating": "N/A", "year": "N/A"}
        self.poster_cache[title] = details
        return details

    def feedback(self, user_id, title, action):
        movie = next((m for m in movies if m["original_title"] == title), None)
        if not movie:
            return "Movie not found."
        reward = 1 if action == "like" else -1
        self.bandit.update(movie.get("id"), reward)
        self.record_interaction(user_id or "demo_user", movie, action)
        return f"Recorded {action} feedback for {title}. Bandit reward estimate updated."

    def record_interaction(self, user_id, movie, action):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO interactions
                (user_id, movie_id, movie_title, action, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    movie.get("id", 0),
                    movie.get("original_title", ""),
                    action,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def analytics(self, user_id):
        user_id = user_id or "demo_user"
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                """
                SELECT action, COUNT(*) FROM interactions
                WHERE user_id = ?
                GROUP BY action
                ORDER BY COUNT(*) DESC
                """,
                (user_id,),
            ).fetchall()
        if not rows:
            return "No interactions recorded yet."
        return "\n".join(f"{action}: {count}" for action, count in rows)


recommender = MovieRecommender()


def get_recommendations(user_id, selected_movie, context, algorithm):
    if not selected_movie:
        return [], "Please select a movie."
    recommendations = recommender.recommend(selected_movie, context, algorithm)
    if not recommendations:
        return [], "No recommendations found."

    gallery = []
    lines = []
    for rank, movie in enumerate(recommendations, start=1):
        details = recommender.fetch_details(movie["original_title"])
        if details["poster"]:
            gallery.append((details["poster"], movie["original_title"]))
        genres = ", ".join(movie.get("genres", "").split()) or "movie features"
        lines.append(f"{rank}. {movie['original_title']} — {genres}")
        recommender.record_interaction(user_id or "demo_user", movie, "view")
    return gallery, "\n".join(lines)


with gr.Blocks(title="Movie Recommender System") as demo:
    gr.Markdown("# 🎬 Movie Recommender System")
    gr.Markdown("Content-based, context-aware and feedback-adaptive movie recommendations.")

    with gr.Tab("Recommendations"):
        with gr.Row():
            with gr.Column():
                user_id = gr.Textbox(value="demo_user", label="User ID")
                selected_movie = gr.Dropdown(choices=movie_titles, label="Select a movie", filterable=True)
                context = gr.Dropdown(
                    choices=["general", "evening", "family", "romantic", "action"],
                    value="general",
                    label="Context",
                )
                algorithm = gr.Dropdown(
                    choices=["Content-Based", "Context-Aware", "Hybrid", "Multi-Armed Bandit"],
                    value="Hybrid",
                    label="Recommendation Method",
                )
                recommend_button = gr.Button("Recommend", variant="primary")
            with gr.Column():
                gallery = gr.Gallery(label="Recommended Movies", columns=5, rows=1, object_fit="contain")
                explanation = gr.Textbox(label="Recommendation Details", lines=8)

        recommend_button.click(
            get_recommendations,
            inputs=[user_id, selected_movie, context, algorithm],
            outputs=[gallery, explanation],
        )

    with gr.Tab("Feedback"):
        feedback_user = gr.Textbox(value="demo_user", label="User ID")
        feedback_movie = gr.Dropdown(choices=movie_titles, label="Movie")
        feedback_action = gr.Radio(choices=["like", "dislike"], value="like", label="Feedback")
        feedback_button = gr.Button("Submit Feedback")
        feedback_status = gr.Textbox(label="Status")
        feedback_button.click(
            recommender.feedback,
            inputs=[feedback_user, feedback_movie, feedback_action],
            outputs=feedback_status,
        )

    with gr.Tab("Analytics"):
        analytics_user = gr.Textbox(value="demo_user", label="User ID")
        analytics_button = gr.Button("Show Analytics")
        analytics_output = gr.Textbox(label="Interaction Summary", lines=8)
        analytics_button.click(
            recommender.analytics,
            inputs=analytics_user,
            outputs=analytics_output,
        )


if __name__ == "__main__":
    demo.launch()
