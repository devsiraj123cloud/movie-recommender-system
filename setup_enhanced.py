"""
Setup script for the Enhanced Movie Recommender System
Run this script to set up everything you need.
"""

import subprocess
import sys
import os
import sqlite3
from datetime import datetime

def install_requirements():
    """Install all required packages"""
    print("📦 Installing required packages...")
    try:
        # Try to find the requirements file in the movie-recommender-system directory
        req_file = "movie-recommender-system/enhanced_requirements.txt"
        if not os.path.exists(req_file):
            req_file = "enhanced_requirements.txt"
        
        if not os.path.exists(req_file):
            # Install packages manually if requirements file is missing
            packages = [
                "gradio>=4.0.0",
                "pandas>=1.5.0", 
                "numpy>=1.21.0",
                "scikit-learn>=1.0.0",
                "requests>=2.28.0",
                "matplotlib>=3.5.0",
                "seaborn>=0.11.0"
            ]
            
            for package in packages:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def setup_database():
    """Set up the SQLite database"""
    print("🗄️ Setting up database...")
    try:
        conn = sqlite3.connect("movie_recommender.db")
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
        
        # Insert some dummy data for testing
        cursor.execute('''
            INSERT OR IGNORE INTO user_interactions 
            (user_id, movie_id, action, timestamp, session_id)
            VALUES 
            ('demo_user', 1, 'view', '2024-01-01 10:00:00', 'session1'),
            ('demo_user', 2, 'like', '2024-01-01 10:05:00', 'session1'),
            ('demo_user', 3, 'view', '2024-01-01 10:10:00', 'session1'),
            ('demo_user', 4, 'dislike', '2024-01-01 10:15:00', 'session1')
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database setup complete!")
        return True
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False

def check_pickle_files():
    """Check if pickle files exist"""
    print("🥒 Checking pickle files...")
    required_files = ['movie_dict.pkl', 'similarity.pkl']
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Found {file}")
        else:
            print(f"❌ Missing {file}")
            print("   Run generate_pickles.py first to create these files")
            return False
    
    return True

def create_demo_config():
    """Create a demo configuration file"""
    print("⚙️ Creating demo configuration...")
    
    config = {
        "omdb_api_key": "325fade6",
        "default_user": "demo_user",
        "algorithms": ["hybrid", "content_based", "contextual", "multi_armed_bandit"],
        "contexts": ["general", "evening", "family", "romantic", "action"],
        "reinforcement_learning": {
            "epsilon": 0.1,
            "learning_rate": 0.001,
            "decay_rate": 0.95
        }
    }
    
    try:
        import json
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Configuration file created!")
        return True
    except Exception as e:
        print(f"❌ Error creating config: {e}")
        return False

def main():
    """Main setup function"""
    print("🎬 Enhanced Movie Recommender System Setup")
    print("=" * 50)
    
    steps = [
        ("Installing packages", install_requirements),
        ("Setting up database", setup_database),
        ("Checking pickle files", check_pickle_files),
        ("Creating configuration", create_demo_config)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            print("Please fix the issue and run setup again.")
            return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Make sure your pickle files are generated: python generate_pickles.py")
    print("2. Run the enhanced app: python enhanced_app.py")
    print("3. Open the provided URL in your browser")
    print("\nFeatures available:")
    print("• 🎯 Multiple recommendation algorithms")
    print("• 🤖 Reinforcement Learning (Multi-Armed Bandit)")
    print("• 🎨 Contextual recommendations")
    print("• 📊 User analytics and A/B testing")
    print("• 📝 Movie rating and feedback system")
    print("• 🧪 Algorithm comparison tools")
    
    return True

if __name__ == "__main__":
    main()
