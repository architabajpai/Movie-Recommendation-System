"""
train.py
End-to-end pipeline: generate data → fit model → evaluate → save.

Usage:
    python train.py                    # use defaults
    python train.py --n 5200           # generate n movies
    python train.py --data movies.csv  # use an existing CSV
"""

import argparse
import os
import sys
import time

import pandas as pd

# ensure project root is on the path when run from sub-dirs
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data.generate_data import generate
from models.recommender import MovieRecommender
from utils.evaluation import evaluate_recommender


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train the Movie Recommendation System")
    p.add_argument("--data",   default=None,  help="Path to existing movies.csv (skips generation)")
    p.add_argument("--n",      type=int, default=5200, help="Number of synthetic movies to generate")
    p.add_argument("--out",    default=os.path.join(ROOT, "saved_model.pkl"), help="Where to save the model")
    p.add_argument("--alpha",  type=float, default=0.15, help="Blend weight for vote signal (0–1)")
    p.add_argument("--no-eval", action="store_true", help="Skip evaluation step")
    return p.parse_args()


# ── pipeline ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1. Load / generate data
    if args.data and os.path.exists(args.data):
        print(f"[✓] Loading dataset from {args.data}")
        df = pd.read_csv(args.data)
    else:
        csv_path = os.path.join(ROOT, "data", "movies.csv")
        if os.path.exists(csv_path):
            print(f"[✓] Found existing dataset at {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            print(f"[~] Generating {args.n:,} synthetic movies …")
            df = generate(n=args.n, out_path=csv_path)

    print(f"\n{'─'*55}")
    print(f"  Dataset: {len(df):,} movies, {df.shape[1]} columns")
    print(f"{'─'*55}\n")

    # 2. Fit
    model = MovieRecommender(blend_alpha=args.alpha)
    model.fit(df)

    # 3. Quick sanity check
    print("\n── Quick demo ──────────────────────────────────────────")
    sample_title = df["title"].iloc[0]
    try:
        recs = model.recommend(sample_title, n=5)
        print(f"Top-5 recommendations for  »{sample_title}«\n")
        print(recs[["title", "genres", "score"]].to_string(index=False))
    except Exception as exc:
        print(f"  Demo skipped: {exc}")

    # 4. Optional evaluation
    if not args.no_eval:
        print("\n── Evaluation ──────────────────────────────────────────")
        evaluate_recommender(model, df, n_queries=100)

    # 5. Save
    print()
    model.save(args.out)
    print("\nDone ✓")


if __name__ == "__main__":
    main()
