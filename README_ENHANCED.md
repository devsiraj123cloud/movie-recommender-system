# 🎬 Enhanced Movie Recommender System

An advanced movie recommendation system powered by Machine Learning, Reinforcement Learning, and modern UI/UX principles.

## 🚀 Features

### Core Recommendation Algorithms
- **Hybrid Recommendation**: Combines content-based, collaborative filtering, and contextual recommendations
- **Content-Based Filtering**: Uses movie features (genre, director, cast, plot) for similarity matching
- **Contextual Recommendations**: Adapts suggestions based on time, mood, and user context
- **Multi-Armed Bandit**: Uses reinforcement learning to optimize recommendations over time

### Advanced Capabilities
- **🤖 Reinforcement Learning**: Learn from user interactions and improve recommendations
- **📊 User Analytics**: Track user behavior and preferences with detailed analytics
- **🧪 A/B Testing**: Compare different recommendation algorithms
- **📝 Rating System**: Users can rate movies to improve future recommendations
- **🎨 Modern UI**: Beautiful, responsive interface with multiple tabs and features
- **💾 Persistent Storage**: SQLite database for user interactions and ratings

### Smart Features
- **Explanation Engine**: Explains why movies were recommended
- **Context Awareness**: Considers time of day, mood, and social context
- **Real-time Learning**: Adapts to user feedback immediately
- **Performance Monitoring**: Tracks algorithm performance and user engagement

## 📋 Requirements

```
gradio>=4.0.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
requests>=2.28.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 🛠️ Installation & Setup

### Step 1: Install Dependencies
```bash
# Activate your virtual environment
& "C:\Users\siraj\Desktop\New folder (5)\movierec-env\Scripts\Activate.ps1"

# Install new packages
pip install matplotlib seaborn
```

### Step 2: Run Setup Script
```bash
python setup_enhanced.py
```

### Step 3: Generate Data Files (if not already done)
```bash
python generate_pickles.py
```

### Step 4: Launch the Enhanced App
```bash
python enhanced_app.py
```

## 🎯 How to Use

### 1. Get Recommendations Tab
- **User ID**: Enter your unique identifier
- **Context/Mood**: Select your current mood or context
- **Algorithm**: Choose the recommendation algorithm
- **Base Movie**: Optionally select a movie you like for similar recommendations

### 2. Rate Movies Tab
- Search for movies and rate them 1-5 stars
- View detailed movie information
- Your ratings improve future recommendations

### 3. Analytics Tab
- View your interaction history
- See charts of your movie preferences
- Track your engagement patterns

### 4. A/B Testing Tab
- Compare different recommendation algorithms
- See which performs better for your preferences
- Help improve the system

## 🤖 Reinforcement Learning Features

### Multi-Armed Bandit
- **Epsilon-Greedy Strategy**: Balances exploration vs exploitation
- **Adaptive Learning**: Epsilon decreases over time
- **Performance Tracking**: Monitors which movies get better user feedback

### Contextual Bandit
- **Context-Aware**: Considers user situation (time, mood, social setting)
- **Dynamic Weights**: Updates recommendation weights based on context
- **Personalization**: Learns individual user patterns

### Deep Q-Learning (Simplified)
- **State Representation**: User profile + current context
- **Action Selection**: Choose movies to recommend
- **Experience Replay**: Learns from past interactions

## 📊 Analytics & Insights

### User Metrics
- **Interaction Counts**: Views, likes, dislikes, ratings
- **Preference Patterns**: Genre preferences, time-based patterns
- **Engagement Score**: Overall system interaction level

### Algorithm Performance
- **Click-Through Rate**: How often recommendations are selected
- **Rating Correlation**: How well predictions match user ratings
- **Diversity Score**: Variety in recommendations

### A/B Testing Results
- **Statistical Significance**: Compare algorithm performance
- **User Satisfaction**: Track user preference for different approaches
- **Conversion Metrics**: Measure recommendation effectiveness

## 🔧 Technical Architecture

### Backend Components
1. **MovieRecommenderSystem**: Main recommendation engine
2. **RLRecommendationEngine**: Reinforcement learning algorithms
3. **Database Layer**: SQLite for persistent storage
4. **API Integration**: OMDb for movie metadata and posters

### Frontend Components
1. **Gradio Interface**: Modern, responsive web UI
2. **Multiple Tabs**: Organized feature access
3. **Interactive Charts**: Real-time analytics visualization
4. **Responsive Design**: Works on desktop and mobile

### Data Flow
```
User Input → Context Analysis → Algorithm Selection → 
Movie Retrieval → Poster Fetching → UI Display → 
User Feedback → RL Update → Performance Tracking
```

## 🎨 Customization Options

### Adding New Algorithms
1. Create new recommendation function in `MovieRecommenderSystem`
2. Add to algorithm dropdown choices
3. Update the recommendation handler

### Extending Context Types
1. Add new context options in the dropdown
2. Update `contextual_recommend` function
3. Add context mapping in RL module

### Custom RL Algorithms
1. Implement new algorithm in `reinforcement_learning.py`
2. Add to `RLRecommendationEngine`
3. Update the selection interface

## 📈 Performance Optimization

### Caching Strategy
- Movie details cached to reduce API calls
- Similarity matrix precomputed and stored
- User profiles cached in memory

### Database Optimization
- Indexed queries for fast user lookups
- Batch updates for better performance
- Periodic cleanup of old data

### RL Optimization
- Efficient numpy operations
- Memory management for experience replay
- Adaptive learning rates

## 🔍 Debugging & Monitoring

### Debug Features
- Console logging for API calls
- Database query logging
- RL algorithm state tracking

### Performance Monitoring
- Response time tracking
- Memory usage monitoring
- User engagement metrics

## 🚀 Future Enhancements

### Planned Features
1. **Deep Learning Integration**: Neural collaborative filtering
2. **Social Recommendations**: Friend-based suggestions
3. **Real-time Streaming**: Live recommendation updates
4. **Mobile App**: Native mobile interface
5. **Voice Interface**: Voice-controlled recommendations

### Advanced ML Features
1. **Transformer Models**: For better text understanding
2. **Graph Neural Networks**: For complex user-item relationships
3. **Federated Learning**: Privacy-preserving recommendations
4. **AutoML**: Automated algorithm selection

## 📞 Support & Troubleshooting

### Common Issues
1. **Missing Pickle Files**: Run `generate_pickles.py`
2. **API Key Issues**: Check OMDb API key validity
3. **Database Errors**: Delete `movie_recommender.db` and run setup again
4. **Package Issues**: Reinstall requirements

### Getting Help
- Check console output for error messages
- Verify all files are in correct locations
- Ensure virtual environment is activated
- Review database schema if needed

## 📝 License & Credits

This project demonstrates advanced recommendation system concepts including:
- Machine Learning algorithms
- Reinforcement Learning techniques  
- Modern web UI/UX design
- Database management
- API integration
- Performance optimization

Created as an educational and practical example of modern recommendation systems.

---

**Enjoy exploring the world of AI-powered movie recommendations! 🎬✨**
