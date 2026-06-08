"""
generate_data.py
Generates a synthetic dataset of 5000+ movies for the recommendation system.
Run this once to create movies.csv before using the recommender.
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# ── reference pools ──────────────────────────────────────────────────────────

GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "Western",
    "Biography", "History", "Sport", "War", "Family",
]

DIRECTORS = [
    "Christopher Nolan", "Steven Spielberg", "Quentin Tarantino",
    "Martin Scorsese", "James Cameron", "Ridley Scott",
    "David Fincher", "Peter Jackson", "Tim Burton", "Wes Anderson",
    "Denis Villeneuve", "Alfonso Cuaron", "Alejandro Inarritu",
    "Guillermo del Toro", "Francis Ford Coppola", "Stanley Kubrick",
    "Clint Eastwood", "Ron Howard", "JJ Abrams", "Zack Snyder",
    "Gore Verbinski", "Bryan Singer", "Sam Raimi", "Guy Ritchie",
    "Paul Thomas Anderson", "Darren Aronofsky", "Joel Coen",
    "Ethan Coen", "Robert Zemeckis", "Tony Scott",
]

ACTORS = [
    "Leonardo DiCaprio", "Tom Hanks", "Brad Pitt", "Morgan Freeman",
    "Robert De Niro", "Al Pacino", "Johnny Depp", "Matt Damon",
    "Christian Bale", "Ryan Gosling", "Denzel Washington", "Will Smith",
    "Tom Cruise", "Harrison Ford", "Keanu Reeves", "Hugh Jackman",
    "Scarlett Johansson", "Meryl Streep", "Cate Blanchett", "Natalie Portman",
    "Jennifer Lawrence", "Emma Stone", "Charlize Theron", "Angelina Jolie",
    "Joaquin Phoenix", "Jared Leto", "Benedict Cumberbatch", "Jake Gyllenhaal",
    "Mark Wahlberg", "Samuel L. Jackson", "Bruce Willis", "Sylvester Stallone",
    "Arnold Schwarzenegger", "Liam Neeson", "Russell Crowe", "Geoffrey Rush",
    "Anthony Hopkins", "Daniel Day-Lewis", "Dustin Hoffman", "Jack Nicholson",
]

KEYWORDS = [
    "love", "war", "survival", "friendship", "betrayal", "redemption",
    "revenge", "heist", "magic", "robot", "alien", "zombie", "vampire",
    "detective", "spy", "superhero", "journey", "family", "prison",
    "corruption", "time travel", "dystopia", "space", "ocean", "forest",
    "city", "desert", "snow", "underground", "chase", "mystery", "cult",
    "conspiracy", "treasure", "prophecy", "dragon", "sword", "gun",
    "hacker", "virus", "clone", "mutation", "revolution", "election",
]

TITLE_PREFIXES = [
    "The", "A", "Dark", "Last", "Lost", "Final", "Hidden", "Broken",
    "Silent", "Eternal", "Forgotten", "Rise of", "Fall of", "Return of",
    "Shadow of", "Dawn of", "Age of", "War of", "Kingdom of", "Edge of",
]

TITLE_NOUNS = [
    "Storm", "Legend", "Dragon", "Kingdom", "Empire", "Alliance",
    "Warrior", "Hunter", "Guardian", "Sentinel", "Phoenix", "Titan",
    "Raven", "Wolf", "Ghost", "Phantom", "Specter", "Eclipse",
    "Horizon", "Frontier", "Destiny", "Origin", "Chronicle",
    "Prophecy", "Covenant", "Revelation", "Redemption", "Judgment",
    "Sacrifice", "Labyrinth", "Odyssey", "Infinity", "Abyss",
]

# ── helper functions ─────────────────────────────────────────────────────────

def random_title(used: set) -> str:
    for _ in range(200):
        title = f"{random.choice(TITLE_PREFIXES)} {random.choice(TITLE_NOUNS)}"
        suffix = random.choice(["", f" {random.randint(2, 5)}", f": Part {random.randint(1,3)}"])
        title = (title + suffix).strip()
        if title not in used:
            used.add(title)
            return title
    # fallback
    t = f"Movie {len(used)+1}"
    used.add(t)
    return t


def random_genres(n_min=1, n_max=4) -> str:
    return " ".join(random.sample(GENRES, random.randint(n_min, n_max)))


def random_cast(n=3) -> str:
    return " ".join(random.sample(ACTORS, n))


def random_keywords(n_min=3, n_max=7) -> str:
    return " ".join(random.sample(KEYWORDS, random.randint(n_min, n_max)))


def random_overview() -> str:
    templates = [
        "A {adj} {char} must {action} before {threat} destroys everything they hold dear.",
        "When {char} discovers {secret}, they are drawn into a {adj} world of {setting}.",
        "Set in {setting}, the film follows {char} as they navigate {challenge} and uncover {secret}.",
        "In a {adj} tale of {theme}, {char} confronts {threat} that will test their very soul.",
        "{char} embarks on a {adj} journey through {setting} to stop {threat} once and for all.",
    ]
    adjs      = ["desperate","thrilling","heart-breaking","mysterious","action-packed","chilling"]
    chars     = ["a lone hero","two unlikely allies","a hardened detective","a young prodigy","an exiled warrior"]
    actions   = ["stop the villain","save their family","uncover the truth","escape captivity","unite the factions"]
    threats   = ["an ancient evil","a shadowy organisation","a catastrophic event","a deadly conspiracy","an unstoppable force"]
    secrets   = ["a long-buried secret","a shocking truth","an ancient prophecy","a hidden identity","a forgotten past"]
    settings  = ["a post-apocalyptic city","a magical realm","a crime-ridden metropolis","a distant galaxy","a war-torn land"]
    challenges= ["political intrigue","impossible odds","a personal crisis","moral dilemmas","unexpected betrayals"]
    themes    = ["love and loss","courage and sacrifice","justice and revenge","identity and belonging","power and corruption"]

    t = random.choice(templates)
    return t.format(
        adj=random.choice(adjs), char=random.choice(chars),
        action=random.choice(actions), threat=random.choice(threats),
        secret=random.choice(secrets), setting=random.choice(settings),
        challenge=random.choice(challenges), theme=random.choice(themes),
    )


# ── generate dataset ─────────────────────────────────────────────────────────

def generate(n: int = 5200, out_path: str = "movies.csv") -> pd.DataFrame:
    used_titles: set = set()
    records = []

    for i in range(1, n + 1):
        records.append({
            "movie_id":   i,
            "title":      random_title(used_titles),
            "genres":     random_genres(),
            "director":   random.choice(DIRECTORS),
            "cast":       random_cast(),
            "keywords":   random_keywords(),
            "overview":   random_overview(),
            "vote_average": round(random.uniform(4.0, 9.5), 1),
            "vote_count":   random.randint(50, 50000),
            "popularity":   round(random.uniform(1.0, 100.0), 2),
            "release_year": random.randint(1970, 2024),
            "runtime":      random.randint(75, 200),
            "language":     random.choice(["en", "fr", "de", "es", "ja", "ko", "hi"]),
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[✓] Saved {len(df):,} movies → {out_path}")
    return df


if __name__ == "__main__":
    generate(n=5200, out_path=os.path.join(os.path.dirname(__file__), "movies.csv"))
