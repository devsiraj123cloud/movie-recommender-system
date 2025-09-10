# 🎬 AI-Powered Movie Recommender System

An intelligent movie recommendation system built with Python and Gradio, featuring multiple recommendation algorithms, user analytics, and a modern web interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Gradio](https://img.shields.io/badge/gradio-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### Core Functionality
- **Multiple Recommendation Algorithms**: Content-based, Hybrid, Contextual, and Multi-Armed Bandit
- **Real-time Movie Posters**: Integration with OMDb API for movie posters and details
- **Interactive Web Interface**: Built with Gradio for seamless user experience
- **User Analytics Dashboard**: Track user interactions and preferences
- **A/B Testing Framework**: Compare different recommendation algorithms

### Advanced Features
- **Reinforcement Learning**: Multi-Armed Bandit algorithm for personalized recommendations
- **User Profiling**: Dynamic user preference learning based on interactions
- **Context-Aware Recommendations**: Time-based and preference-based contextual suggestions
- **SQLite Database**: Persistent storage for user interactions and ratings
- **Recommendation Explanations**: AI-generated explanations for each recommendation

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movie-recommender-system.git
   cd movie-recommender-system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv movierec-env
   
   # On Windows
   movierec-env\Scripts\activate
   
   # On macOS/Linux
   source movierec-env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r enhanced_requirements.txt
   ```

4. **Set up the database and generate movie data**
   ```bash
   python setup_enhanced.py
   ```

5. **Get OMDb API Key** (Free)
   - Visit [OMDb API](http://www.omdbapi.com/apikey.aspx)
   - Sign up for a free API key
   - Update the API key in the code files

## 🎯 Usage

### Simple Movie Recommender
For a basic movie recommendation experience:
```bash
python gradio_app.py
```

### Enhanced Movie Recommender
For the full-featured experience with analytics and multiple algorithms:
```bash
python simple_Demo.py
```

### Advanced System
For the complete system with all features:
```bash
python enhanced_app.py
```

## 📱 Applications

### 1. Basic Recommender (`gradio_app.py`)
- Simple movie search and recommendations
- Movie poster display
- Clean, minimalist interface

### 2. Enhanced Recommender (`simple_Demo.py`)
- **Main Tab**: Core recommendation functionality
- **Analytics Tab**: User interaction statistics and visualizations
- **A/B Testing Tab**: Algorithm performance comparison
- **Settings Tab**: User preferences and system configuration

### 3. Advanced System (`enhanced_app.py`)
- All features from enhanced recommender
- Advanced machine learning algorithms
- Comprehensive user profiling
- Real-time recommendation explanations

## 🛠️ Technical Architecture

### Components
- **Frontend**: Gradio web interface
- **Backend**: Python with scikit-learn for ML algorithms
- **Database**: SQLite for user data persistence
- **API Integration**: OMDb API for movie metadata
- **ML Algorithms**: TF-IDF, Cosine Similarity, Multi-Armed Bandit

### File Structure
```
movie-recommender-system/
├── gradio_app.py              # Simple Gradio app
├── simple_Demo.py             # Enhanced multi-tab app
├── enhanced_app.py            # Advanced system
├── reinforcement_learning.py  # Multi-Armed Bandit implementation
├── setup_enhanced.py          # Database and data setup
├── generate_pickles.py        # Movie data preprocessing
├── movie_dict.pkl            # Preprocessed movie data
├── similarity.pkl            # Precomputed similarity matrix
├── enhanced_requirements.txt  # Python dependencies
└── README.md                 # This file
```

### Algorithm Details

#### Content-Based Filtering
Uses TF-IDF vectorization and cosine similarity to find movies with similar content characteristics.

#### Multi-Armed Bandit
Implements epsilon-greedy exploration strategy to balance between exploiting known preferences and exploring new recommendations.

#### Contextual Recommendations
Considers user context (time of day, previous interactions) to provide more relevant suggestions.

#### Hybrid Approach
Combines multiple algorithms with weighted scoring for improved accuracy.

## 📊 Analytics Features

- **User Interaction Tracking**: Monitor clicks, ratings, and viewing patterns
- **Algorithm Performance**: Compare recommendation accuracy across different methods
- **Visualization Dashboard**: Interactive charts and graphs
- **A/B Testing**: Systematic comparison of recommendation strategies

## 🔧 Configuration

### API Configuration
Update the OMDb API key in the configuration section of each app file:
```python
OMDB_API_KEY = "your_api_key_here"
```

### Database Configuration
The system uses SQLite by default. Database location can be configured:
```python
DB_PATH = "movie_recommender.db"
```

## 📈 Performance

- **Response Time**: < 2 seconds for recommendations
- **Database**: Handles 10,000+ user interactions efficiently
- **Scalability**: Designed for single-user deployment, easily adaptable for multi-user
- **Memory Usage**: ~100MB for movie data and similarity matrices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
pip install -r enhanced_requirements.txt
```

**2. Missing Pickle Files**
```bash
python generate_pickles.py
```

**3. Database Issues**
```bash
python setup_enhanced.py
```

**4. API Key Issues**
- Ensure you have a valid OMDb API key
- Check your internet connection
- Verify API key is correctly set in the code

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Movie Data**: Based on TMDB movie dataset
- **OMDb API**: For movie posters and metadata
- **Gradio**: For the intuitive web interface
- **scikit-learn**: For machine learning algorithms

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Search existing issues in the GitHub repository
3. Create a new issue with detailed description

## 🔮 Future Enhancements

- [ ] Deep learning-based recommendations
- [ ] Real-time collaborative filtering
- [ ] Mobile app development
- [ ] Integration with streaming platforms
- [ ] Social features and sharing
- [ ] Advanced natural language processing for reviews

---

⭐ **Star this repository if you found it helpful!** ⭐
