# 🎬 Movie Recommendation System

Content-based recommendation engine using **cosine similarity** and **TF-IDF feature vectorisation** — built with Python, Pandas, and Scikit-learn.

---

## Project Structure

```
movie-recommender/
├── data/
│   ├── generate_data.py      # synthetic 5,200-movie dataset generator
│   └── movies.csv            # auto-created on first run
├── models/
│   └── recommender.py        # core MovieRecommender class
├── utils/
│   ├── preprocessor.py       # cleaning, soup-building, TF-IDF vectorisation
│   └── evaluation.py         # genre overlap, diversity, coverage metrics
├── train.py                  # end-to-end training pipeline
├── app.py                    # interactive CLI
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model  (generates data + saves saved_model.pkl)
python train.py

# 3. Launch the interactive CLI
python app.py
```

---

## Architecture

### Feature Engineering (`utils/preprocessor.py`)

Each movie's raw text fields are combined into a **"soup"** string with importance weighting:

| Field    | Weight | Why |
|----------|--------|-----|
| Director | ×3     | Strong stylistic signal |
| Genres   | ×2     | Primary content category |
| Cast     | ×2     | Star-based affinity |
| Keywords | ×2     | Thematic tags |
| Overview | ×1     | Descriptive context |

Multi-word names (e.g. `Christopher Nolan`) are underscored into single tokens (`christopher_nolan`) so the TF-IDF tokeniser treats them atomically.

### Vectorisation

`TfidfVectorizer` with:
- `ngram_range=(1, 2)` — captures bigrams like `action adventure`
- `sublinear_tf=True` — dampens very frequent terms
- `min_df=2`, `max_df=0.95` — removes ultra-rare and ubiquitous tokens
- `stop_words="english"` — removes noise words

### Similarity (`models/recommender.py`)

```
sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

Final ranking blends cosine similarity with a normalised vote signal:

```
blended_score = (1 - α) × cosine_sim  +  α × vote_score_norm
```

Default `α = 0.15` keeps content similarity dominant while breaking ties by popularity.

---

## CLI Commands

```
recommend  <title>                  top-10 recommendations
recommend  <title> -n <k>           top-k recommendations
info       <title>                  full movie metadata
search     <query>                  find titles containing query
filter     <title> [options]        filtered recommendations
  --genre  <Genre>
  --lang   <ISO code, e.g. en>
  --from   <year>
  --to     <year>
stats                               dataset summary
help                                show all commands
quit                                exit
```

---

## Evaluation Metrics (`utils/evaluation.py`)

| Metric | Description |
|--------|-------------|
| Genre Overlap | Mean Jaccard similarity of genres between query & recs |
| Director Hit Rate | Fraction of recs sharing the same director |
| Intra-list Diversity | Mean pairwise genre distance within each list |
| Mean Rec Score | Average blended cosine score across all queries |
| Catalogue Coverage | Fraction of catalogue that ever gets recommended |

---

## Using as a Library

```python
import pandas as pd
from models.recommender import MovieRecommender

df = pd.read_csv("data/movies.csv")

model = MovieRecommender(blend_alpha=0.15)
model.fit(df)

recs = model.recommend("The Dark Frontier", n=10)
print(recs)

# with filters
recs = model.recommend(
    "The Dark Frontier", n=10,
    filter_genre="Action",
    filter_language="en",
    min_year=2000,
)

# save / load
model.save("saved_model.pkl")
model = MovieRecommender.load("saved_model.pkl")
```

---

## Dataset

The synthetic dataset contains **5,200 movies** with:

- `movie_id`, `title`, `genres`, `director`, `cast`, `keywords`, `overview`
- `vote_average`, `vote_count`, `popularity`, `release_year`, `runtime`, `language`

Replace `data/movies.csv` with a real dataset (e.g. TMDB 5000 from Kaggle) — the pipeline is drop-in compatible as long as the column names match.


*Deployed app:*
https://movie-recommendation-system-3z5wzyappugxvaappvvdmvdt.streamlit.app/
