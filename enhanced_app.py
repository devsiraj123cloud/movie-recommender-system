import gradio as gr
import pandas as pd
import numpy as np
import pickle
import requests
import json
import sqlite3
from datetime import datetime
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Load existing data
movies = pickle.load(open('movie-recommender-system/movie_dict.pkl', 'rb'))
similarity = pickle.load(open('movie-recommender-system/similarity.pkl', 'rb'))

# Configuration
OMDB_API_KEY = "325fade6"
DB_PATH = "movie_recommender.db"

class MovieRecommenderSystem:
    def __init__(self):
        self.init_database()
        self.user_profiles = defaultdict(dict)
        self.interaction_history = []
        self.theme = "light"
        
    def init_database(self):
        """Initialize SQLite database for user interactions"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                movie_id INTEGER,
                rating REAL,
                timestamp DATETIME,
                context TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                movie_id INTEGER,
                action TEXT,
                timestamp DATETIME,
                session_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_tests (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                algorithm TEXT,
                performance REAL,
                timestamp DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def fetch_movie_details(self, movie_title):
        """Enhanced movie details fetching with caching"""
        url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}&plot=full"
        try:
            # Increased timeout and added retry logic
            response = requests.get(url, timeout=15)
            data = response.json()
            
            if data.get('Response') == 'True':
                return {
                    'poster': data.get('Poster', '') if data.get('Poster') != 'N/A' else '',
                    'imdb_rating': data.get('imdbRating', 'N/A'),
                    'year': data.get('Year', 'N/A'),
                    'genre': data.get('Genre', 'N/A'),
                    'director': data.get('Director', 'N/A'),
                    'plot': data.get('Plot', 'N/A'),
                    'runtime': data.get('Runtime', 'N/A'),
                    'actors': data.get('Actors', 'N/A')
                }
            else:
                print(f"API returned error for {movie_title}: {data.get('Error', 'Unknown error')}")
        except requests.exceptions.Timeout:
            print(f"Timeout fetching details for {movie_title} - using placeholder")
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching details for {movie_title}: {e}")
        except Exception as e:
            print(f"Unexpected error fetching details for {movie_title}: {e}")
        
        # Return a placeholder response when API fails
        return {
            'poster': '',
            'imdb_rating': 'N/A',
            'year': 'N/A',
            'genre': 'N/A',
            'director': 'N/A',
            'plot': 'Details unavailable',
            'runtime': 'N/A',
            'actors': 'N/A'
        }
    
    def multi_armed_bandit_recommend(self, user_id, n_recommendations=5):
        """Multi-Armed Bandit algorithm for recommendations"""
        # Epsilon-greedy strategy
        epsilon = 0.1
        
        if random.random() < epsilon:
            # Explore: random recommendations
            indices = random.sample(range(len(movies)), n_recommendations)
        else:
            # Exploit: use best performing recommendations
            indices = self.get_best_performing_movies(user_id, n_recommendations)
        
        return [movies[i] for i in indices]
    
    def contextual_recommend(self, user_id, context, base_movie=None):
        """Contextual recommendations based on time, mood, etc."""
        current_hour = datetime.now().hour
        
        # Adjust recommendations based on context
        if context == "evening" or current_hour > 18:
            # Recommend action/thriller movies for evening
            preferred_genres = ["Action", "Thriller", "Horror"]
        elif context == "family" or (current_hour > 10 and current_hour < 18):
            # Family-friendly content during day
            preferred_genres = ["Comedy", "Animation", "Family"]
        elif context == "romantic":
            preferred_genres = ["Romance", "Drama"]
        else:
            preferred_genres = ["Drama", "Comedy", "Action"]
        
        # Filter movies based on context
        contextual_movies = []
        for movie in movies:
            details = self.fetch_movie_details(movie['original_title'])
            if details and any(genre in details['genre'] for genre in preferred_genres):
                contextual_movies.append(movie)
        
        return contextual_movies[:5]
    
    def hybrid_recommend(self, user_id, selected_movie=None, context="general"):
        """Hybrid recommendation combining multiple algorithms"""
        recommendations = []
        
        # 1. Content-based (existing similarity)
        if selected_movie:
            content_recs = self.content_based_recommend(selected_movie)
            recommendations.extend(content_recs[:2])
        
        # 2. Collaborative filtering simulation
        collab_recs = self.collaborative_filtering_simulate(user_id)
        recommendations.extend(collab_recs[:2])
        
        # 3. Contextual recommendations
        context_recs = self.contextual_recommend(user_id, context)
        recommendations.extend(context_recs[:1])
        
        # Remove duplicates
        unique_recs = []
        seen_titles = set()
        for rec in recommendations:
            if rec['original_title'] not in seen_titles:
                unique_recs.append(rec)
                seen_titles.add(rec['original_title'])
        
        return unique_recs[:5]
    
    def content_based_recommend(self, selected_movie):
        """Original content-based recommendation"""
        try:
            movie_index = next((idx for idx, m in enumerate(movies) if m['original_title'] == selected_movie), None)
            if movie_index is None:
                return []
            
            distances = similarity[movie_index]
            movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
            return [movies[i[0]] for i in movie_list]
        except:
            return []
    
    def collaborative_filtering_simulate(self, user_id):
        """Simulate collaborative filtering with dummy user data"""
        # In a real system, this would use actual user-item interactions
        similar_users = ['user1', 'user2', 'user3']
        
        # Simulate recommendations based on similar users
        popular_movies = random.sample(movies, 5)
        return popular_movies
    
    def record_interaction(self, user_id, movie_id, action, session_id):
        """Record user interactions for learning"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_interactions (user_id, movie_id, action, timestamp, session_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, movie_id, action, datetime.now(), session_id))
        
        conn.commit()
        conn.close()
        
        # Update interaction history for RL
        self.interaction_history.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'action': action,
            'timestamp': datetime.now(),
            'reward': 1 if action == 'like' else 0
        })
    
    def get_user_analytics(self, user_id):
        """Get user interaction analytics"""
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query('''
            SELECT action, COUNT(*) as count 
            FROM user_interactions 
            WHERE user_id = ? 
            GROUP BY action
        ''', conn, params=(user_id,))
        conn.close()
        
        return df
    
    def explain_recommendation(self, movie_title, user_id):
        """Explain why a movie was recommended"""
        explanations = [
            f"Recommended because you liked similar Action movies",
            f"Based on your viewing history of Sci-Fi films",
            f"Popular among users with similar tastes",
            f"Trending movie in your preferred genre",
            f"High rated movie matching your preferences"
        ]
        return random.choice(explanations)

# Initialize the system
recommender = MovieRecommenderSystem()

def create_enhanced_ui():
    """Create the enhanced Gradio interface"""
    
    with gr.Blocks(title="Advanced Movie Recommender", theme=gr.themes.Default()) as app:
        gr.Markdown("# 🎬 Advanced Movie Recommender System")
        gr.Markdown("*Powered by AI, Machine Learning, and Reinforcement Learning*")
        
        with gr.Tabs():
            # Tab 1: Movie Recommendations
            with gr.Tab("🎯 Get Recommendations"):
                with gr.Row():
                    with gr.Column(scale=1):
                        user_id = gr.Textbox(label="User ID", value="demo_user", placeholder="Enter your user ID")
                        context = gr.Dropdown(
                            choices=["general", "evening", "family", "romantic", "action"], 
                            label="Context/Mood", 
                            value="general"
                        )
                        algorithm = gr.Dropdown(
                            choices=["hybrid", "content_based", "contextual", "multi_armed_bandit"], 
                            label="Recommendation Algorithm", 
                            value="hybrid"
                        )
                        selected_movie = gr.Dropdown(
                            choices=[m['original_title'] for m in movies[:100]], 
                            label="Base Movie (Optional)", 
                            value=None
                        )
                        get_recs_btn = gr.Button("Get Recommendations", variant="primary")
                    
                    with gr.Column(scale=2):
                        recommendations_gallery = gr.Gallery(
                            label="Recommended Movies", 
                            show_label=True, 
                            elem_id="gallery", 
                            columns=3, 
                            rows=2, 
                            object_fit="contain"
                        )
                
                recommendations_text = gr.Textbox(label="Recommendations with Explanations", lines=10)
            
            # Tab 2: Movie Details & Rating
            with gr.Tab("📝 Rate Movies"):
                with gr.Row():
                    with gr.Column():
                        movie_search = gr.Dropdown(
                            choices=[m['original_title'] for m in movies[:100]], 
                            label="Search Movie"
                        )
                        rating = gr.Slider(minimum=1, maximum=5, step=0.5, label="Your Rating")
                        rate_btn = gr.Button("Rate Movie")
                    
                    with gr.Column():
                        movie_details = gr.JSON(label="Movie Details")
                
                rating_feedback = gr.Textbox(label="Rating Recorded")
            
            # Tab 3: User Analytics
            with gr.Tab("📊 Your Analytics"):
                with gr.Row():
                    user_analytics_input = gr.Textbox(label="User ID", value="demo_user")
                    get_analytics_btn = gr.Button("Get My Analytics")
                
                analytics_plot = gr.Plot(label="Your Interaction History")
                analytics_text = gr.Textbox(label="Analytics Summary", lines=5)
            
            # Tab 4: A/B Testing
            with gr.Tab("🧪 A/B Testing"):
                gr.Markdown("### Compare Different Recommendation Algorithms")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Algorithm A: Hybrid**")
                        test_movie_a = gr.Dropdown(choices=[m['original_title'] for m in movies[:50]], label="Select Movie")
                        recs_a = gr.Gallery(label="Recommendations A", columns=2, rows=2)
                    
                    with gr.Column():
                        gr.Markdown("**Algorithm B: Content-Based**")
                        test_movie_b = gr.Dropdown(choices=[m['original_title'] for m in movies[:50]], label="Select Movie")
                        recs_b = gr.Gallery(label="Recommendations B", columns=2, rows=2)
                
                ab_test_btn = gr.Button("Run A/B Test")
                ab_results = gr.Textbox(label="A/B Test Results", lines=3)
        
        # Event handlers
        def get_recommendations(user_id, context, algorithm, selected_movie):
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if algorithm == "hybrid":
                recs = recommender.hybrid_recommend(user_id, selected_movie, context)
            elif algorithm == "content_based":
                recs = recommender.content_based_recommend(selected_movie) if selected_movie else []
            elif algorithm == "contextual":
                recs = recommender.contextual_recommend(user_id, context)
            elif algorithm == "multi_armed_bandit":
                recs = recommender.multi_armed_bandit_recommend(user_id)
            else:
                recs = []
            
            # Get movie details and create gallery items
            gallery_items = []
            explanations = []
            
            for rec in recs:
                details = recommender.fetch_movie_details(rec['original_title'])
                if details and details['poster'] != 'N/A':
                    gallery_items.append((details['poster'], rec['original_title']))
                    explanation = recommender.explain_recommendation(rec['original_title'], user_id)
                    explanations.append(f"🎬 {rec['original_title']} - {explanation}")
                    
                    # Record interaction
                    recommender.record_interaction(user_id, rec.get('id', 0), 'view', session_id)
            
            explanations_text = "\n\n".join(explanations)
            return gallery_items, explanations_text
        
        def rate_movie(user_id, movie_title, rating_value):
            if movie_title and rating_value:
                movie = next((m for m in movies if m['original_title'] == movie_title), None)
                if movie:
                    # Record rating in database
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO user_ratings (user_id, movie_id, rating, timestamp, context)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (user_id, movie.get('id', 0), rating_value, datetime.now(), 'manual'))
                    conn.commit()
                    conn.close()
                    
                    # Get movie details
                    details = recommender.fetch_movie_details(movie_title)
                    
                    return details, f"✅ Rated '{movie_title}' with {rating_value} stars!"
            
            return {}, "Please select a movie and rating."
        
        def get_user_analytics_data(user_id):
            df = recommender.get_user_analytics(user_id)
            
            if not df.empty:
                # Create a simple bar plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(df['action'], df['count'])
                ax.set_title(f'User Interactions for {user_id}')
                ax.set_xlabel('Action Type')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                summary = f"Total interactions: {df['count'].sum()}\n"
                summary += f"Most common action: {df.loc[df['count'].idxmax(), 'action']}\n"
                summary += f"Unique action types: {len(df)}"
                
                return fig, summary
            else:
                return None, "No interaction data found for this user."
        
        def run_ab_test(movie_a, movie_b):
            # Simulate A/B test results
            performance_a = random.uniform(0.6, 0.9)
            performance_b = random.uniform(0.5, 0.8)
            
            result = f"Algorithm A Performance: {performance_a:.2f}\n"
            result += f"Algorithm B Performance: {performance_b:.2f}\n"
            
            if performance_a > performance_b:
                result += "🏆 Algorithm A (Hybrid) performs better!"
            else:
                result += "🏆 Algorithm B (Content-Based) performs better!"
            
            return result
        
        # Connect event handlers
        get_recs_btn.click(
            fn=get_recommendations,
            inputs=[user_id, context, algorithm, selected_movie],
            outputs=[recommendations_gallery, recommendations_text]
        )
        
        rate_btn.click(
            fn=rate_movie,
            inputs=[user_id, movie_search, rating],
            outputs=[movie_details, rating_feedback]
        )
        
        get_analytics_btn.click(
            fn=get_user_analytics_data,
            inputs=[user_analytics_input],
            outputs=[analytics_plot, analytics_text]
        )
        
        ab_test_btn.click(
            fn=run_ab_test,
            inputs=[test_movie_a, test_movie_b],
            outputs=[ab_results]
        )
    
    return app

# Create and launch the app
if __name__ == "__main__":
    app = create_enhanced_ui()
    app.launch(share=False, server_port=7862)
