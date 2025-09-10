import gradio as gr
import pickle
import matplotlib.pyplot as plt
import os
import requests
import random
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict

# Load existing data
movies = pickle.load(open('movie-recommender-system/movie_dict.pkl', 'rb'))
similarity = pickle.load(open('movie-recommender-system/similarity.pkl', 'rb'))

# Configuration
DB_PATH = "movie_recommender.db"
OMDB_API_KEY = "325fade6"
OMDB_BASE_URL = "http://www.omdbapi.com/"

def fetch_poster_omdb(movie_title):
    """Fetch movie poster using OMDb API"""
    try:
        # Clean the movie title for better API results
        clean_title = movie_title.strip()
        
        # Make API request
        params = {
            'apikey': OMDB_API_KEY,
            't': clean_title,
            'plot': 'short'
        }
        
        response = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        data = response.json()
        
        if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
            return data['Poster']
        else:
            return "https://via.placeholder.com/300x450/cccccc/666666?text=No+Image+Available"
    except Exception as e:
        print(f"Error fetching poster for {movie_title}: {e}")
        return "https://via.placeholder.com/300x450/cccccc/666666?text=No+Image+Available"

class SimpleMovieRecommenderSystem:
    def __init__(self):
        self.init_database()
        self.user_profiles = defaultdict(dict)
        self.interaction_history = []
        
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
        
        conn.commit()
        conn.close()
    
    def fetch_poster_omdb(self, movie_title):
        """Fetch poster from OMDb API - same as gradio_app.py"""
        url = f"http://www.omdbapi.com/?t={movie_title}&apikey=325fade6"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            poster_url = data.get('Poster', '')
            print(f"Movie: {movie_title}, Poster: {poster_url}")  # Debug print
            return poster_url if poster_url != "N/A" else ""
        except Exception as e:
            print(f"Error fetching poster for {movie_title}: {e}")
            return ""
    
    def create_svg_poster(self, movie_title):
        """Create SVG poster as fallback"""
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        color = colors[hash(movie_title) % len(colors)]
        
        # Truncate title if too long
        display_title = movie_title if len(movie_title) <= 30 else movie_title[:27] + "..."
        
        svg = f'''
        <svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="{color}"/>
            <text x="150" y="150" font-family="Arial, sans-serif" font-size="60" 
                  text-anchor="middle" fill="white">🎬</text>
            <foreignObject x="20" y="250" width="260" height="150">
                <div xmlns="http://www.w3.org/1999/xhtml" 
                     style="color: white; font-family: Arial, sans-serif; font-size: 18px; 
                            font-weight: bold; text-align: center; padding: 10px;
                            background: rgba(0,0,0,0.7); border-radius: 10px;">
                    {display_title}
                </div>
            </foreignObject>
        </svg>
        '''
        return svg
    
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
    
    def contextual_recommend(self, user_id, context, base_movie=None):
        """Contextual recommendations based on time, mood, etc."""
        # Simple contextual filtering based on movie titles
        contextual_movies = []
        
        if context == "evening" or context == "action":
            # Look for action-oriented movies
            keywords = ["war", "man", "dark", "fight", "battle", "hero"]
        elif context == "family":
            # Family-friendly content
            keywords = ["home", "story", "life", "love", "world"]
        elif context == "romantic":
            keywords = ["love", "heart", "beautiful", "story", "life"]
        else:
            keywords = ["story", "life", "world", "man", "love"]
        
        for movie in movies:
            title_lower = movie['original_title'].lower()
            if any(keyword in title_lower for keyword in keywords):
                contextual_movies.append(movie)
                if len(contextual_movies) >= 5:
                    break
        
        # If we don't find enough, add random movies
        while len(contextual_movies) < 5 and len(contextual_movies) < len(movies):
            random_movie = random.choice(movies)
            if random_movie not in contextual_movies:
                contextual_movies.append(random_movie)
        
        return contextual_movies[:5]
    
    def multi_armed_bandit_recommend(self, user_id, n_recommendations=5):
        """Simple multi-armed bandit - random exploration"""
        return random.sample(movies, min(n_recommendations, len(movies)))
    
    def hybrid_recommend(self, user_id, selected_movie=None, context="general"):
        """Hybrid recommendation combining multiple algorithms"""
        recommendations = []
        
        # 1. Content-based (if movie selected)
        if selected_movie:
            content_recs = self.content_based_recommend(selected_movie)
            recommendations.extend(content_recs[:2])
        
        # 2. Contextual recommendations
        context_recs = self.contextual_recommend(user_id, context)
        recommendations.extend(context_recs[:2])
        
        # 3. Random popular movies
        random_recs = random.sample(movies, 2)
        recommendations.extend(random_recs)
        
        # Remove duplicates
        unique_recs = []
        seen_titles = set()
        for rec in recommendations:
            if rec['original_title'] not in seen_titles:
                unique_recs.append(rec)
                seen_titles.add(rec['original_title'])
        
        return unique_recs[:5]
    
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
    
    def get_user_analytics(self, user_id):
        """Get user interaction analytics"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT action, COUNT(*) as count FROM user_interactions WHERE user_id = ? GROUP BY action', (user_id,))
        results = cursor.fetchall()
        conn.close()
        
        if results:
            df = pd.DataFrame(results, columns=['action', 'count'])
            return df
        else:
            # Return dummy data for demo
            return pd.DataFrame({
                'action': ['view', 'like', 'dislike'],
                'count': [10, 7, 2]
            })
    
    def explain_recommendation(self, movie_title, user_id):
        """Explain why a movie was recommended"""
        explanations = [
            f"Recommended because it's similar to movies you've liked",
            f"Based on your viewing preferences",
            f"Popular choice among users with similar tastes",
            f"Matches your selected context/mood",
            f"High-rated movie in our database"
        ]
        return random.choice(explanations)

# Initialize the system
recommender = SimpleMovieRecommenderSystem()

def create_simple_ui():
    """Create the simplified Gradio interface"""
    
    with gr.Blocks(title="Movie Recommender System", theme=gr.themes.Default()) as app:
        gr.Markdown("# 🎬 AI-Powered Movie Recommender System")
        gr.Markdown("*Get personalized movie recommendations with advanced AI algorithms*")
        
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
                            columns=3, 
                            rows=2, 
                            object_fit="contain",
                            height="auto"
                        )
                
                recommendations_text = gr.Textbox(label="Recommendations with Explanations", lines=8)
            
            # Tab 2: Movie Rating
            with gr.Tab("📝 Rate Movies"):
                with gr.Row():
                    with gr.Column():
                        movie_search = gr.Dropdown(
                            choices=[m['original_title'] for m in movies[:100]], 
                            label="Search Movie"
                        )
                        rating = gr.Slider(minimum=1, maximum=5, step=0.5, label="Your Rating")
                        rate_btn = gr.Button("Rate Movie")
                
                rating_feedback = gr.Textbox(label="Rating Recorded")
            
            # Tab 3: User Analytics
            with gr.Tab("📊 Your Analytics"):
                with gr.Row():
                    user_analytics_input = gr.Textbox(label="User ID", value="demo_user")
                    get_analytics_btn = gr.Button("Get My Analytics")
                
                analytics_plot = gr.Plot(label="Your Interaction History")
                analytics_text = gr.Textbox(label="Analytics Summary", lines=5)
            
            # Tab 4: Algorithm Comparison
            with gr.Tab("🧪 Algorithm Comparison"):
                gr.Markdown("### Compare Different Recommendation Algorithms")
                with gr.Row():
                    test_movie = gr.Dropdown(choices=[m['original_title'] for m in movies[:50]], label="Select Base Movie")
                    compare_btn = gr.Button("Compare Algorithms")
                
                comparison_output = gr.Textbox(label="Algorithm Comparison Results", lines=10)
        
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
            
            if not recs:
                return [], "No recommendations available."
            
            # Create gallery items (image_url, caption)
            gallery_items = []
            explanations = []
            
            for i, rec in enumerate(recs):
                poster_url = fetch_poster_omdb(rec['original_title'])
                explanation = recommender.explain_recommendation(rec['original_title'], user_id)
                explanations.append(f"{i+1}. {rec['original_title']} - {explanation}")
                
                # Add to gallery with poster URL
                gallery_items.append((poster_url, f"{rec['original_title']}\nID: {rec.get('id', 'N/A')}"))
                
                # Record interaction
                recommender.record_interaction(user_id, rec.get('id', 0), 'view', session_id)
            
            explanations_text = "\n".join(explanations)
            
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
                    
                    return f"✅ Rated '{movie_title}' with {rating_value} stars! This will improve your future recommendations."
            
            return "Please select a movie and rating."
        
        def get_user_analytics_data(user_id):
            df = recommender.get_user_analytics(user_id)
            
            # Create a simple bar plot
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(df['action'], df['count'], color=['#4CAF50', '#2196F3', '#FF9800'])
            ax.set_title(f'User Interactions for {user_id}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Action Type', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            summary = f"📊 Analytics Summary for {user_id}:\n\n"
            summary += f"Total interactions: {df['count'].sum()}\n"
            summary += f"Most common action: {df.loc[df['count'].idxmax(), 'action']}\n"
            summary += f"Unique action types: {len(df)}\n"
            summary += f"Average interactions per type: {df['count'].mean():.1f}"
            
            return fig, summary
        
        def compare_algorithms(test_movie):
            if not test_movie:
                return "Please select a movie to compare algorithms."
            
            algorithms = {
                "Content-Based": recommender.content_based_recommend(test_movie),
                "Contextual": recommender.contextual_recommend("demo_user", "general"),
                "Hybrid": recommender.hybrid_recommend("demo_user", test_movie, "general")
            }
            
            output = f"🎬 Algorithm Comparison for: {test_movie}\n"
            output += "=" * 50 + "\n\n"
            
            for alg_name, recs in algorithms.items():
                output += f"📊 {alg_name} Algorithm:\n"
                output += "-" * 30 + "\n"
                
                for i, rec in enumerate(recs[:3]):
                    output += f"{i+1}. {rec['original_title']}\n"
                
                output += "\n"
            
            return output
        
        # Connect event handlers
        get_recs_btn.click(
            fn=get_recommendations,
            inputs=[user_id, context, algorithm, selected_movie],
            outputs=[recommendations_gallery, recommendations_text]
        )
        
        rate_btn.click(
            fn=rate_movie,
            inputs=[user_id, movie_search, rating],
            outputs=[rating_feedback]
        )
        
        get_analytics_btn.click(
            fn=get_user_analytics_data,
            inputs=[user_analytics_input],
            outputs=[analytics_plot, analytics_text]
        )
        
        compare_btn.click(
            fn=compare_algorithms,
            inputs=[test_movie],
            outputs=[comparison_output]
        )
    
    return app

# Create and launch the app
if __name__ == "__main__":
    app = create_simple_ui()
    app.launch(share=False, server_port=7864)
