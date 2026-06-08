"""
utils/evaluation.py
Evaluation helpers for the content-based recommender.

Metrics
-------
* Genre overlap (proxy for relevance)  – fraction of recommendations that share
  at least one genre with the query movie.
* Director hit rate                    – fraction that share the same director.
* Intra-list diversity                 – average pairwise genre-Jaccard distance
  inside the recommendation list.
* Mean recommendation score            – average blended cosine score returned.
* Coverage                             – fraction of the catalogue that ever
  appears in at least one recommendation list.
"""

from __future__ import annotations

import random
import numpy as np
import pandas as pd

from models.recommender import MovieRecommender


# ── metric helpers ────────────────────────────────────────────────────────────

def genre_set(genre_str: str) -> set[str]:
    return set(str(genre_str).lower().split())


def genre_overlap(query_genres: str, rec_genres: str) -> float:
    q, r = genre_set(query_genres), genre_set(rec_genres)
    if not q or not r:
        return 0.0
    return len(q & r) / len(q | r)   # Jaccard similarity


def intra_list_diversity(rec_genres: list[str]) -> float:
    """Mean pairwise Jaccard *distance* (1 - similarity) within the list."""
    if len(rec_genres) < 2:
        return 0.0
    scores = []
    for i in range(len(rec_genres)):
        for j in range(i + 1, len(rec_genres)):
            scores.append(1.0 - genre_overlap(rec_genres[i], rec_genres[j]))
    return float(np.mean(scores))


# ── main evaluation function ──────────────────────────────────────────────────

def evaluate_recommender(
    model: MovieRecommender,
    df: pd.DataFrame,
    n_queries: int = 100,
    top_k: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """
    Sample `n_queries` movies at random and compute aggregate quality metrics.

    Returns a dict of metric_name → value and prints a formatted table.
    """
    random.seed(seed)
    sample_titles = random.sample(list(df["title"]), min(n_queries, len(df)))

    genre_overlaps: list[float] = []
    director_hits:  list[float] = []
    diversities:    list[float] = []
    mean_scores:    list[float] = []
    recommended_ids: set        = set()

    for title in sample_titles:
        info = model.get_movie_info(title)
        if info is None:
            continue
        try:
            recs = model.recommend(title, n=top_k)
        except Exception:
            continue

        if recs.empty:
            continue

        q_genres   = str(info.get("genres", ""))
        q_director = str(info.get("director", "")).lower()

        # genre overlap (mean Jaccard over recs)
        genre_overlaps.append(
            np.mean([genre_overlap(q_genres, g) for g in recs["genres"].fillna("")])
        )

        # director hit rate
        if "director" in recs.columns:
            hit = (recs["director"].str.lower() == q_director).mean()
            director_hits.append(float(hit))

        # intra-list diversity
        diversities.append(intra_list_diversity(list(recs["genres"].fillna(""))))

        # mean score
        if "score" in recs.columns:
            mean_scores.append(float(recs["score"].mean()))

        # coverage tracking (use title as id)
        recommended_ids.update(recs["title"].tolist())

    catalogue_size = len(df)
    coverage = len(recommended_ids) / catalogue_size if catalogue_size else 0.0

    results = {
        "queries_evaluated":    len(genre_overlaps),
        "genre_overlap_mean":   float(np.mean(genre_overlaps))   if genre_overlaps else 0,
        "director_hit_rate":    float(np.mean(director_hits))     if director_hits  else 0,
        "intra_list_diversity": float(np.mean(diversities))       if diversities    else 0,
        "mean_rec_score":       float(np.mean(mean_scores))       if mean_scores    else 0,
        "catalogue_coverage":   coverage,
    }

    # pretty-print
    print(f"\n{'Metric':<28} {'Value':>10}")
    print("─" * 40)
    for k, v in results.items():
        if k == "queries_evaluated":
            print(f"  {k:<26} {int(v):>10}")
        else:
            print(f"  {k:<26} {v:>10.4f}")
    print("─" * 40)

    return results
