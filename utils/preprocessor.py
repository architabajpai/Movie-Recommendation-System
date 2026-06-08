"""
utils/preprocessor.py
Feature-engineering and text-vectorization utilities for the movie dataset.
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


# ── text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise_field(text: str) -> str:
    """
    Convert space / comma separated values into underscored tokens so that
    multi-word names like 'Christopher Nolan' become 'christopher_nolan' and
    don't get split by the TF-IDF tokeniser.
    """
    if not isinstance(text, str):
        return ""
    # replace commas and multiple spaces with a single space
    text = re.sub(r"[,]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    # join individual tokens into underscore form
    return " ".join("_".join(t.split()) for t in text.split(",") if t)


def build_soup(row: pd.Series) -> str:
    """
    Combine several feature columns into one weighted 'soup' string.
    Fields are repeated proportionally to their importance weight.
    """
    genres   = clean_text(row.get("genres", ""))
    director = "_".join(clean_text(row.get("director", "")).split())
    cast     = " ".join(
        "_".join(a.strip().split())
        for a in str(row.get("cast", "")).split(" ")
        if a.strip()
    )
    keywords = clean_text(row.get("keywords", ""))
    overview = clean_text(row.get("overview", ""))

    # weights: director × 3, genres × 2, cast × 2, keywords × 2, overview × 1
    parts = (
        [director] * 3
        + genres.split() * 2
        + cast.split() * 2
        + keywords.split() * 2
        + overview.split()
    )
    return " ".join(parts)


# ── preprocessing pipeline ────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw DataFrame and add derived columns needed by the recommender.

    Returns a copy with at minimum:
      - soup            (str) combined feature string
      - vote_score_norm (float 0–1) normalised rating signal
    """
    df = df.copy()

    # fill missing text columns
    for col in ("genres", "director", "cast", "keywords", "overview"):
        df[col] = df[col].fillna("").astype(str)

    # build soup
    df["soup"] = df.apply(build_soup, axis=1)

    # normalise numeric signals (used for optional re-ranking)
    scaler = MinMaxScaler()
    numeric_cols = []
    for col in ("vote_average", "vote_count", "popularity"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            numeric_cols.append(col)

    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        df["vote_score_norm"] = (
            df.get("vote_average", 0) * 0.6
            + df.get("vote_count", 0) * 0.2
            + df.get("popularity", 0) * 0.2
        )
    else:
        df["vote_score_norm"] = 0.0

    return df


# ── vectorisation ─────────────────────────────────────────────────────────────

def build_tfidf_matrix(soups: pd.Series, **kwargs):
    """
    Fit a TF-IDF vectoriser on the soup column and return (matrix, vectoriser).
    Extra kwargs are forwarded to TfidfVectorizer.
    """
    defaults = dict(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english",
    )
    defaults.update(kwargs)
    vec = TfidfVectorizer(**defaults)
    mat = vec.fit_transform(soups)
    return mat, vec
