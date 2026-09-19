# 🎬 Movie Recommender System

A single Gradio application combining **content-based filtering, context-aware recommendations, and an epsilon-greedy Multi-Armed Bandit**.

## Flow

TMDB Movies + Credits → Pandas preprocessing → feature extraction → tags → CountVectorizer → movie vectors → cosine similarity → recommendation strategy → OMDb metadata/posters → Gradio → user feedback → bandit update + SQLite history.

## Features

- Content-based filtering using CountVectorizer and cosine similarity.
- Context-aware recommendations using movie genres.
- Hybrid recommendations combining content, context and feedback.
- Epsilon-greedy Multi-Armed Bandit updated by likes/dislikes.
- OMDb integration for posters and IMDb ratings.
- SQLite interaction tracking and analytics.
- One main Gradio UI.

## Project structure

\`\`\`
movie-recommender-system/
├── app.py
├── generate_pickles.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── movie_dict.pkl       # generated locally
└── similarity.pkl       # generated locally
\`\`\`

The previous simple, enhanced, and standalone RL versions have been consolidated into one application. They are no longer part of the main project flow.

## Setup

Put the TMDB files in:

\`\`\`
data/
├── tmdb_5000_movies.csv
└── tmdb_5000_credits.csv
\`\`\`

Install:

\`\`\`bash
python -m venv venv
pip install -r requirements.txt
\`\`\`

Generate model data:

\`\`\`bash
python generate_pickles.py
\`\`\`

Set an OMDb API key as an environment variable:

Windows PowerShell:
\`\`\`powershell
$env:OMDB_API_KEY="YOUR_OMDB_API_KEY"
\`\`\`

macOS/Linux:
\`\`\`bash
export OMDB_API_KEY="YOUR_OMDB_API_KEY"
\`\`\`

Run:

\`\`\`bash
python app.py
\`\`\`

## Recommendation methods

### Content-Based
Combines overview, genres, director, top cast and keywords. CountVectorizer converts the tags into word-count vectors, then cosine similarity finds similar movies.

### Context-Aware
Uses selected context and movie genres to find relevant candidates.

### Hybrid
Combines:
- 60% content similarity
- 25% context match
- 15% learned feedback reward

### Multi-Armed Bandit
Uses epsilon-greedy exploration/exploitation. A like gives +1 reward and a dislike gives -1 reward. The observed average reward is used when selecting candidates.

This is a lightweight educational bandit component, not production-scale RL.

## Accuracy note

The actual preprocessing uses **CountVectorizer**, not TF-IDF.

The earlier simulated collaborative-filtering and simulated A/B-testing implementations were removed so the repository represents one coherent, defensible project.

## Resume description

**AI-Powered Movie Recommender System**
- Developed a content-based movie recommender using movie metadata, CountVectorizer and cosine similarity.
- Built hybrid and context-aware recommendation strategies.
- Integrated an epsilon-greedy Multi-Armed Bandit to adapt recommendations from user feedback.
- Integrated OMDb API and built an interactive Gradio interface.
- Added SQLite-based interaction tracking and analytics.

## License

MIT License.
