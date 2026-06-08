"""
models/recommender.py
Content-based Movie Recommendation Engine using cosine similarity.
"""

from __future__ import annotations

import pickle
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessor import preprocess, build_tfidf_matrix


class MovieRecommender:
    """
    Content-based recommendation engine.

    Flow:
      1. Preprocess raw movie DataFrame (clean + build soup).
      2. Vectorise soup with TF-IDF.
      3. Compute pairwise cosine similarity matrix.
      4. On query, find the k nearest neighbours and (optionally) re-rank
         by a blend of cosine score + normalised vote signal.

    Parameters
    ----------
    blend_alpha : float
        Weight for cosine similarity vs. vote score when re-ranking.
        0 → pure cosine, 1 → pure vote score. Default 0.15.
    """

    def __init__(self, blend_alpha: float = 0.15):
        self.blend_alpha = blend_alpha
        self._df: pd.DataFrame | None = None
        self._sim_matrix: np.ndarray | None = None
        self._title_index: dict[str, int] = {}
        self._fitted = False

    # ── public API ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "MovieRecommender":
        """Preprocess data, vectorise, and compute similarity matrix."""
        t0 = time.time()

        if verbose:
            print(f"[1/3] Preprocessing {len(df):,} movies …")
        self._df = preprocess(df).reset_index(drop=True)

        if verbose:
            print("[2/3] Building TF-IDF feature matrix …")
        mat, _ = build_tfidf_matrix(self._df["soup"])

        if verbose:
            print(f"      Feature matrix shape: {mat.shape}")
            print("[3/3] Computing cosine similarity matrix …")
        self._sim_matrix = cosine_similarity(mat, mat)

        # title → row-index lookup (lowercased for case-insensitive search)
        self._title_index = {
            t.lower(): i for i, t in enumerate(self._df["title"])
        }

        self._fitted = True
        elapsed = time.time() - t0
        if verbose:
            print(f"[✓] Model fitted in {elapsed:.2f}s")
        return self

    def recommend(
        self,
        title: str,
        n: int = 10,
        filter_language: str | None = None,
        filter_genre: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> pd.DataFrame:
        """
        Return the top-n recommendations for a given movie title.

        Parameters
        ----------
        title          : Exact movie title (case-insensitive).
        n              : Number of recommendations to return.
        filter_language: ISO 639-1 code, e.g. 'en'.
        filter_genre   : Genre string (partial match), e.g. 'Action'.
        min_year       : Earliest release year to include.
        max_year       : Latest release year to include.

        Returns
        -------
        pd.DataFrame with columns [title, genres, director, cast,
                                   vote_average, release_year, score]
        """
        self._check_fitted()
        key = title.strip().lower()
        if key not in self._title_index:
            raise ValueError(
                f"Movie '{title}' not found in the dataset. "
                "Try get_closest_titles() for suggestions."
            )

        idx = self._title_index[key]
        sim_scores = self._sim_matrix[idx].copy()

        # blend with vote signal
        vote_norm = self._df["vote_score_norm"].values
        blended = (1 - self.blend_alpha) * sim_scores + self.blend_alpha * vote_norm

        # sort descending, skip the query movie itself
        order = np.argsort(blended)[::-1]
        recs = self._df.iloc[order].copy()
        recs["_score"] = blended[order]
        recs = recs[recs["title"].str.lower() != key]

        # optional filters
        if filter_language:
            recs = recs[recs["language"] == filter_language]
        if filter_genre:
            recs = recs[recs["genres"].str.contains(filter_genre, case=False, na=False)]
        if min_year is not None and "release_year" in recs.columns:
            recs = recs[recs["release_year"] >= min_year]
        if max_year is not None and "release_year" in recs.columns:
            recs = recs[recs["release_year"] <= max_year]

        display_cols = [
            c for c in
            ["title", "genres", "director", "cast", "vote_average",
             "release_year", "_score"]
            if c in recs.columns
        ]
        result = recs[display_cols].head(n).rename(columns={"_score": "score"})
        return result.reset_index(drop=True)

    def get_closest_titles(self, query: str, k: int = 5) -> list[str]:
        """Return up to k titles whose lowercase form contains the query."""
        self._check_fitted()
        q = query.lower()
        matches = [t for t in self._df["title"] if q in t.lower()]
        return matches[:k]

    def get_movie_info(self, title: str) -> pd.Series | None:
        """Return all metadata for a single movie (case-insensitive)."""
        self._check_fitted()
        key = title.strip().lower()
        if key not in self._title_index:
            return None
        return self._df.iloc[self._title_index[key]]

    # ── persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the fitted model to disk."""
        self._check_fitted()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[✓] Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "MovieRecommender":
        """Load a previously saved model from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj)}")
        print(f"[✓] Model loaded ← {path}")
        return obj

    # ── helpers ────────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model is not fitted yet. Call .fit(df) first.")

    @property
    def n_movies(self) -> int:
        return len(self._df) if self._df is not None else 0
