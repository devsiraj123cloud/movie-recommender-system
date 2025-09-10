import numpy as np
import json
from datetime import datetime, timedelta

class MultiArmedBandit:
    """
    Multi-Armed Bandit for movie recommendations using epsilon-greedy strategy
    """
    def __init__(self, n_arms, epsilon=0.1, decay_rate=0.95):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.counts = np.zeros(n_arms)  # Number of times each arm was pulled
        self.values = np.zeros(n_arms)  # Average reward for each arm
        self.total_reward = 0
        self.total_count = 0
        
    def select_arm(self):
        """Select an arm using epsilon-greedy strategy"""
        if np.random.random() > self.epsilon:
            # Exploit: choose the arm with highest average reward
            return np.argmax(self.values)
        else:
            # Explore: choose a random arm
            return np.random.randint(0, self.n_arms)
    
    def update(self, chosen_arm, reward):
        """Update the values based on the reward received"""
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        
        # Update average reward for the chosen arm
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
        
        self.total_reward += reward
        self.total_count += 1
        
        # Decay epsilon over time to reduce exploration
        self.epsilon *= self.decay_rate
    
    def get_best_arms(self, k=5):
        """Get the k best performing arms"""
        return np.argsort(self.values)[-k:][::-1]

class ContextualBandit:
    """
    Contextual Bandit that considers user context (time, mood, etc.)
    """
    def __init__(self, n_arms, n_contexts, learning_rate=0.1):
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.learning_rate = learning_rate
        
        # Weight matrix: [contexts x arms]
        self.weights = np.random.normal(0, 0.1, (n_contexts, n_arms))
        
    def get_context_vector(self, user_context):
        """Convert user context to vector representation"""
        context_map = {
            'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3,
            'happy': 4, 'sad': 5, 'excited': 6, 'calm': 7,
            'alone': 8, 'family': 9, 'friends': 10
        }
        
        context_vector = np.zeros(self.n_contexts)
        for ctx in user_context:
            if ctx in context_map:
                context_vector[context_map[ctx]] = 1.0
        
        return context_vector
    
    def select_arm(self, context_vector):
        """Select arm based on context"""
        # Compute expected rewards for each arm given the context
        expected_rewards = np.dot(context_vector, self.weights)
        return np.argmax(expected_rewards)
    
    def update(self, context_vector, chosen_arm, reward):
        """Update weights based on reward"""
        # Compute prediction error
        predicted_reward = np.dot(context_vector, self.weights[:, chosen_arm])
        error = reward - predicted_reward
        
        # Update weights using gradient descent
        self.weights[:, chosen_arm] += self.learning_rate * error * context_vector

class DeepQLearning:
    """
    Simplified Deep Q-Learning for recommendation system
    """
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = []
        self.memory_size = 2000
        
        # Simple neural network weights (simplified for demo)
        self.weights = {
            'input_hidden': np.random.normal(0, 0.1, (state_size, 64)),
            'hidden_output': np.random.normal(0, 0.1, (64, action_size))
        }
    
    def get_state(self, user_profile, current_context):
        """Convert user profile and context to state vector"""
        # Simplified state representation
        state = np.zeros(self.state_size)
        
        # User preferences (first 10 dimensions)
        if 'genres' in user_profile:
            for i, genre in enumerate(user_profile['genres'][:10]):
                if i < 10:
                    state[i] = 1.0
        
        # Current context (next 5 dimensions)
        context_features = [
            current_context.get('time_of_day', 0),
            current_context.get('day_of_week', 0),
            current_context.get('mood_score', 0),
            current_context.get('social_context', 0),
            current_context.get('device_type', 0)
        ]
        
        state[10:15] = context_features[:5]
        return state
    
    def predict_q_values(self, state):
        """Predict Q-values for all actions given a state"""
        # Forward pass through simple network
        hidden = np.maximum(0, np.dot(state, self.weights['input_hidden']))  # ReLU
        q_values = np.dot(hidden, self.weights['hidden_output'])
        return q_values
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        
        q_values = self.predict_q_values(state)
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        # Simplified training (in practice, use proper backpropagation)
        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            
            target = reward
            if not done:
                target += 0.95 * np.max(self.predict_q_values(next_state))
            
            # Update weights (simplified gradient update)
            q_values = self.predict_q_values(state)
            q_values[action] = target
            
            # Simple weight update (in practice, use proper backpropagation)
            learning_signal = (target - q_values[action]) * self.learning_rate
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Recommendation Engine with RL
class RLRecommendationEngine:
    """
    Main recommendation engine that uses RL algorithms
    """
    def __init__(self, n_movies):
        self.n_movies = n_movies
        self.mab = MultiArmedBandit(n_movies)
        self.contextual_bandit = ContextualBandit(n_movies, 11)  # 11 context features
        self.dqn = DeepQLearning(15, n_movies)  # 15 state features
        
    def get_recommendations_mab(self, n_recommendations=5):
        """Get recommendations using Multi-Armed Bandit"""
        recommendations = []
        for _ in range(n_recommendations):
            arm = self.mab.select_arm()
            recommendations.append(arm)
        return recommendations
    
    def get_recommendations_contextual(self, user_context, n_recommendations=5):
        """Get recommendations using Contextual Bandit"""
        context_vector = self.contextual_bandit.get_context_vector(user_context)
        recommendations = []
        
        for _ in range(n_recommendations):
            arm = self.contextual_bandit.select_arm(context_vector)
            recommendations.append(arm)
            
        return recommendations
    
    def get_recommendations_dqn(self, user_profile, current_context, n_recommendations=5):
        """Get recommendations using Deep Q-Learning"""
        state = self.dqn.get_state(user_profile, current_context)
        recommendations = []
        
        for _ in range(n_recommendations):
            action = self.dqn.choose_action(state)
            recommendations.append(action)
            
        return recommendations
    
    def update_from_feedback(self, algorithm, movie_id, reward, context=None, user_profile=None):
        """Update the RL model based on user feedback"""
        if algorithm == 'mab':
            self.mab.update(movie_id, reward)
        elif algorithm == 'contextual' and context:
            context_vector = self.contextual_bandit.get_context_vector(context)
            self.contextual_bandit.update(context_vector, movie_id, reward)
        elif algorithm == 'dqn' and user_profile and context:
            state = self.dqn.get_state(user_profile, context)
            # In practice, you'd need next_state and done flag
            self.dqn.remember(state, movie_id, reward, state, True)
            self.dqn.replay()
    
    def save_models(self, filepath):
        """Save all models to file"""
        models_data = {
            'mab_counts': self.mab.counts.tolist(),
            'mab_values': self.mab.values.tolist(),
            'mab_epsilon': self.mab.epsilon,
            'contextual_weights': self.contextual_bandit.weights.tolist(),
            'dqn_weights': {k: v.tolist() for k, v in self.dqn.weights.items()},
            'dqn_epsilon': self.dqn.epsilon
        }
        
        with open(filepath, 'w') as f:
            json.dump(models_data, f)
    
    def load_models(self, filepath):
        """Load models from file"""
        try:
            with open(filepath, 'r') as f:
                models_data = json.load(f)
            
            self.mab.counts = np.array(models_data['mab_counts'])
            self.mab.values = np.array(models_data['mab_values'])
            self.mab.epsilon = models_data['mab_epsilon']
            
            self.contextual_bandit.weights = np.array(models_data['contextual_weights'])
            
            self.dqn.weights = {k: np.array(v) for k, v in models_data['dqn_weights'].items()}
            self.dqn.epsilon = models_data['dqn_epsilon']
            
            return True
        except:
            return False
