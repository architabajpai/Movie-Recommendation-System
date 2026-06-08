"""
app.py
Interactive command-line interface for the Movie Recommendation System.

Usage:
    python app.py                          # auto-load or auto-train model
    python app.py --model saved_model.pkl  # explicit model path
    python app.py --data data/movies.csv   # explicit dataset path
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.recommender import MovieRecommender

BANNER = r"""
╔══════════════════════════════════════════════════════╗
║          🎬  Movie Recommendation System             ║
║        Content-Based · Cosine Similarity             ║
╚══════════════════════════════════════════════════════╝
"""

HELP = """
Commands
────────
  recommend  <title>           → top-10 recommendations
  recommend  <title> -n <k>    → top-k recommendations
  info       <title>           → full metadata for a movie
  search     <query>           → find titles containing query
  filter     <title> --genre <G> --lang <L> --from <Y> --to <Y>
                               → filtered recommendations
  stats                        → dataset statistics
  help                         → show this message
  quit / exit                  → exit
"""


# ── load / train model ────────────────────────────────────────────────────────

def load_or_train(model_path: str, data_path: str) -> MovieRecommender:
    if os.path.exists(model_path):
        print(f"Loading model from {model_path} …")
        return MovieRecommender.load(model_path)

    print(f"No saved model found at '{model_path}'. Training from scratch …\n")
    if not os.path.exists(data_path):
        from data.generate_data import generate
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        generate(n=5200, out_path=data_path)

    df = pd.read_csv(data_path)
    model = MovieRecommender()
    model.fit(df)
    model.save(model_path)
    return model


# ── command handlers ──────────────────────────────────────────────────────────

def cmd_recommend(model: MovieRecommender, args: list[str]):
    if not args:
        print("  Usage: recommend <title> [-n <k>]")
        return
    n = 10
    if "-n" in args:
        idx = args.index("-n")
        try:
            n = int(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
        except (IndexError, ValueError):
            print("  Invalid -n value.")
            return
    title = " ".join(args)
    try:
        recs = model.recommend(title, n=n)
        print(f"\n  Top-{n} recommendations for  »{title}«\n")
        for i, row in recs.iterrows():
            score_str = f"{row['score']:.4f}" if "score" in row else ""
            year_str  = f"({int(row['release_year'])})" if "release_year" in row else ""
            print(f"  {i+1:>2}. {row['title']} {year_str}")
            print(f"      Genres: {row.get('genres','—')}  |  "
                  f"Director: {row.get('director','—')}  |  Score: {score_str}")
    except ValueError as e:
        print(f"\n  ⚠  {e}")
        suggestions = model.get_closest_titles(title)
        if suggestions:
            print(f"  Did you mean? {suggestions}")


def cmd_filter(model: MovieRecommender, args: list[str]):
    """recommend with optional filters: --genre --lang --from --to"""
    genre = lang = None
    min_year = max_year = None
    title_parts = []
    i = 0
    while i < len(args):
        if args[i] == "--genre" and i + 1 < len(args):
            genre = args[i + 1]; i += 2
        elif args[i] == "--lang" and i + 1 < len(args):
            lang = args[i + 1]; i += 2
        elif args[i] == "--from" and i + 1 < len(args):
            try: min_year = int(args[i + 1])
            except ValueError: pass
            i += 2
        elif args[i] == "--to" and i + 1 < len(args):
            try: max_year = int(args[i + 1])
            except ValueError: pass
            i += 2
        else:
            title_parts.append(args[i]); i += 1

    title = " ".join(title_parts)
    if not title:
        print("  Usage: filter <title> [--genre G] [--lang L] [--from Y] [--to Y]")
        return
    try:
        recs = model.recommend(
            title, n=10,
            filter_language=lang,
            filter_genre=genre,
            min_year=min_year,
            max_year=max_year,
        )
        active = [f"genre={genre}" if genre else "",
                  f"lang={lang}"   if lang  else "",
                  f"from={min_year}" if min_year else "",
                  f"to={max_year}"   if max_year  else ""]
        active = [a for a in active if a]
        fstr = ", ".join(active) if active else "none"
        print(f"\n  Filtered recommendations for »{title}«  (filters: {fstr})\n")
        if recs.empty:
            print("  No results match the filters.")
            return
        for i, row in recs.iterrows():
            print(f"  {i+1:>2}. {row['title']}  ({row.get('release_year','?')})")
            print(f"      {row.get('genres','—')}  |  {row.get('director','—')}")
    except ValueError as e:
        print(f"\n  ⚠  {e}")


def cmd_info(model: MovieRecommender, args: list[str]):
    title = " ".join(args)
    info  = model.get_movie_info(title)
    if info is None:
        print(f"\n  ⚠  '{title}' not found.")
        suggestions = model.get_closest_titles(title)
        if suggestions:
            print(f"  Did you mean? {suggestions}")
        return
    print(f"\n  ── {info['title']} ──────────────────────────")
    for field in ("genres","director","cast","keywords","release_year","runtime",
                  "language","vote_average","vote_count","overview"):
        val = info.get(field, "—")
        print(f"  {field:<14}: {val}")


def cmd_search(model: MovieRecommender, args: list[str]):
    query   = " ".join(args)
    results = model.get_closest_titles(query, k=10)
    if not results:
        print(f"  No titles containing '{query}' found.")
    else:
        print(f"\n  Titles matching '{query}':")
        for t in results:
            print(f"    • {t}")


def cmd_stats(model: MovieRecommender):
    df = model._df
    if df is None:
        print("  Model not fitted.")
        return
    print(f"\n  Total movies   : {len(df):,}")
    print(f"  Feature columns: {list(df.columns)}")
    if "genres" in df.columns:
        all_genres = " ".join(df["genres"].fillna("")).split()
        from collections import Counter
        top = Counter(all_genres).most_common(5)
        print(f"  Top genres     : {', '.join(f'{g}({c})' for g,c in top)}")
    if "release_year" in df.columns:
        print(f"  Year range     : {int(df['release_year'].min())} – {int(df['release_year'].max())}")


# ── main REPL ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation CLI")
    parser.add_argument("--model", default=os.path.join(ROOT, "saved_model.pkl"))
    parser.add_argument("--data",  default=os.path.join(ROOT, "data", "movies.csv"))
    args = parser.parse_args()

    print(BANNER)
    model = load_or_train(args.model, args.data)
    print(f"\n  Loaded {model.n_movies:,} movies. Type 'help' to see commands.\n")

    while True:
        try:
            raw = input("  🎬 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()
        rest  = parts[1:]

        if cmd in ("quit", "exit", "q"):
            print("  Goodbye!")
            break
        elif cmd == "help":
            print(HELP)
        elif cmd == "recommend":
            cmd_recommend(model, rest)
        elif cmd == "filter":
            cmd_filter(model, rest)
        elif cmd == "info":
            cmd_info(model, rest)
        elif cmd == "search":
            cmd_search(model, rest)
        elif cmd == "stats":
            cmd_stats(model)
        else:
            print(f"  Unknown command '{cmd}'. Type 'help' for options.")

        print()


if __name__ == "__main__":
    main()
