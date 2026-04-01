"""
eda.py — Exploratory Data Analysis for RKGCN datasets.

Generates statistics, distributions, and saves plots for:
  - MovieLens-1M (datasets/MovieLens-1M/)
  - Book-Crossing (datasets/Book-Crossing/)

Usage:
  python eda.py --dataset movie
  python eda.py --dataset book
  python eda.py --dataset both

Output:
  Prints statistics to console and saves plots to eda_outputs/{dataset}/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ============================================================
# Common Helpers
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


# ============================================================
# MovieLens-1M EDA
# ============================================================

def eda_movielens(data_dir, output_dir):
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS: MovieLens-1M")
    print("=" * 60)
    ensure_dir(output_dir)

    # --- Load Data ---
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.dat"), sep="::", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"], engine="python",
    )
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.dat"), sep="::", header=None,
        names=["movie_id", "title", "genres"], engine="python", encoding="latin-1",
    )
    users = pd.read_csv(
        os.path.join(data_dir, "users.dat"), sep="::", header=None,
        names=["user_id", "gender", "age", "occupation", "zip"], engine="python",
    )

    # ----- 1. Basic Statistics -----
    print("\n--- Basic Statistics ---")
    print(f"Total ratings:       {len(ratings):,}")
    print(f"Unique users:        {ratings.user_id.nunique():,}")
    print(f"Unique movies:       {ratings.movie_id.nunique():,}")
    print(f"Total movies in DB:  {len(movies):,}")
    print(f"Total users in DB:   {len(users):,}")
    density = len(ratings) / (ratings.user_id.nunique() * ratings.movie_id.nunique()) * 100
    print(f"Density:             {density:.2f}%")
    print(f"Sparsity:            {100 - density:.2f}%")
    print(f"Avg ratings/user:    {ratings.groupby('user_id').size().mean():.1f}")
    print(f"Avg ratings/movie:   {ratings.groupby('movie_id').size().mean():.1f}")
    print(f"Rating range:        {ratings.rating.min()} – {ratings.rating.max()}")
    print(f"Mean rating:         {ratings.rating.mean():.2f}")
    print(f"Median rating:       {ratings.rating.median():.1f}")

    # ----- 2. Rating Distribution -----
    print("\n--- Rating Distribution ---")
    rating_counts = ratings.rating.value_counts().sort_index()
    for r, c in rating_counts.items():
        pct = c / len(ratings) * 100
        bar = "█" * int(pct)
        print(f"  {r} star: {c:>7,} ({pct:5.1f}%) {bar}")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    ax.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor="white", width=0.7)
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("MovieLens-1M: Rating Distribution", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xticks([1, 2, 3, 4, 5])
    save_fig(fig, os.path.join(output_dir, "rating_distribution.png"))

    # ----- 3. Ratings per User -----
    ratings_per_user = ratings.groupby("user_id").size()
    print(f"\n--- Ratings per User ---")
    print(f"  Min:    {ratings_per_user.min()}")
    print(f"  Max:    {ratings_per_user.max():,}")
    print(f"  Mean:   {ratings_per_user.mean():.1f}")
    print(f"  Median: {ratings_per_user.median():.1f}")
    print(f"  Std:    {ratings_per_user.std():.1f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratings_per_user, bins=50, color="#3498db", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of Ratings", fontsize=12)
    ax.set_ylabel("Number of Users", fontsize=12)
    ax.set_title("MovieLens-1M: Ratings per User", fontsize=14, fontweight="bold")
    ax.axvline(ratings_per_user.mean(), color="#e74c3c", ls="--", label=f"Mean: {ratings_per_user.mean():.0f}")
    ax.axvline(ratings_per_user.median(), color="#2ecc71", ls="--", label=f"Median: {ratings_per_user.median():.0f}")
    ax.legend()
    save_fig(fig, os.path.join(output_dir, "ratings_per_user.png"))

    # ----- 4. Ratings per Movie -----
    ratings_per_movie = ratings.groupby("movie_id").size()
    print(f"\n--- Ratings per Movie ---")
    print(f"  Min:    {ratings_per_movie.min()}")
    print(f"  Max:    {ratings_per_movie.max():,}")
    print(f"  Mean:   {ratings_per_movie.mean():.1f}")
    print(f"  Median: {ratings_per_movie.median():.1f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratings_per_movie, bins=50, color="#9b59b6", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of Ratings", fontsize=12)
    ax.set_ylabel("Number of Movies", fontsize=12)
    ax.set_title("MovieLens-1M: Ratings per Movie", fontsize=14, fontweight="bold")
    ax.axvline(ratings_per_movie.mean(), color="#e74c3c", ls="--", label=f"Mean: {ratings_per_movie.mean():.0f}")
    ax.legend()
    save_fig(fig, os.path.join(output_dir, "ratings_per_movie.png"))

    # ----- 5. Genre Analysis -----
    all_genres = movies["genres"].str.split("|").explode()
    genre_counts = all_genres.value_counts()
    print(f"\n--- Genre Distribution ---")
    print(f"  Unique genres: {genre_counts.shape[0]}")
    for g, c in genre_counts.items():
        print(f"  {g:<20s} {c:>5,}")

    fig, ax = plt.subplots(figsize=(10, 6))
    genre_counts_sorted = genre_counts.sort_values(ascending=True)
    ax.barh(genre_counts_sorted.index, genre_counts_sorted.values, color="#1abc9c", edgecolor="white")
    ax.set_xlabel("Number of Movies", fontsize=12)
    ax.set_title("MovieLens-1M: Genre Distribution", fontsize=14, fontweight="bold")
    save_fig(fig, os.path.join(output_dir, "genre_distribution.png"))

    # ----- 6. User Demographics -----
    print(f"\n--- User Demographics ---")
    gender_counts = users.gender.value_counts()
    print(f"  Gender: M={gender_counts.get('M', 0):,}, F={gender_counts.get('F', 0):,}")

    age_labels = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
                  45: "45-49", 50: "50-55", 56: "56+"}
    age_counts = users.age.value_counts().sort_index()
    print(f"  Age groups:")
    for a, c in age_counts.items():
        label = age_labels.get(a, str(a))
        print(f"    {label:<12s} {c:>5,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Gender pie
    axes[0].pie(gender_counts.values, labels=gender_counts.index,
                autopct="%1.1f%%", colors=["#3498db", "#e74c3c"], startangle=90)
    axes[0].set_title("Gender Distribution", fontsize=13, fontweight="bold")
    # Age bar
    age_names = [age_labels.get(a, str(a)) for a in age_counts.index]
    axes[1].bar(age_names, age_counts.values, color="#e67e22", edgecolor="white")
    axes[1].set_xlabel("Age Group", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Age Distribution", fontsize=13, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle("MovieLens-1M: User Demographics", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "user_demographics.png"))

    # ----- 7. Temporal Analysis -----
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["year"] = ratings["datetime"].dt.year
    ratings["month"] = ratings["datetime"].dt.to_period("M")
    yearly = ratings.groupby("year").size()
    print(f"\n--- Temporal Analysis ---")
    print(f"  Date range: {ratings.datetime.min().date()} to {ratings.datetime.max().date()}")
    for y, c in yearly.items():
        print(f"  {y}: {c:>7,}")

    fig, ax = plt.subplots(figsize=(8, 5))
    monthly = ratings.groupby("month").size()
    ax.plot(range(len(monthly)), monthly.values, color="#2c3e50", linewidth=1.5)
    ax.fill_between(range(len(monthly)), monthly.values, color="#3498db", alpha=0.3)
    ax.set_xlabel("Time (monthly)", fontsize=12)
    ax.set_ylabel("Number of Ratings", fontsize=12)
    ax.set_title("MovieLens-1M: Rating Activity Over Time", fontsize=14, fontweight="bold")
    step = max(1, len(monthly) // 10)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels([str(monthly.index[i]) for i in range(0, len(monthly), step)], rotation=45)
    save_fig(fig, os.path.join(output_dir, "temporal_analysis.png"))

    # ----- 8. Positive vs Negative (for RKGCN) -----
    n_pos = len(ratings[ratings.rating >= 4])
    n_neg = len(ratings[ratings.rating < 4])
    print(f"\n--- Implicit Feedback (RKGCN Threshold: rating >= 4) ---")
    print(f"  Positive (≥4): {n_pos:,} ({n_pos/len(ratings)*100:.1f}%)")
    print(f"  Negative (<4): {n_neg:,} ({n_neg/len(ratings)*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Positive (≥4)", "Negative (<4)"], [n_pos, n_neg],
           color=["#27ae60", "#e74c3c"], edgecolor="white", width=0.5)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("MovieLens-1M: Positive vs Negative for RKGCN", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, v in enumerate([n_pos, n_neg]):
        ax.text(i, v + 5000, f"{v:,}", ha="center", fontweight="bold")
    save_fig(fig, os.path.join(output_dir, "positive_negative_split.png"))

    print(f"\n  All plots saved to: {output_dir}/")
    return


# ============================================================
# Book-Crossing EDA
# ============================================================

def eda_bookcrossing(data_dir, output_dir):
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS: Book-Crossing")
    print("=" * 60)
    ensure_dir(output_dir)

    # --- Load Data ---
    ratings = pd.read_csv(
        os.path.join(data_dir, "Ratings.csv"), sep=";", encoding="latin-1",
        on_bad_lines="skip",
    )
    ratings.columns = ["user_id", "isbn", "rating"]
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings.dropna(subset=["rating"], inplace=True)
    ratings["rating"] = ratings["rating"].astype(int)

    books = pd.read_csv(
        os.path.join(data_dir, "Books.csv"), sep=";", encoding="latin-1",
        on_bad_lines="skip", low_memory=False,
    )
    books.columns = ["isbn", "title", "author", "year", "publisher"]

    users = pd.read_csv(
        os.path.join(data_dir, "Users.csv"), sep=";", encoding="latin-1",
        on_bad_lines="skip",
    )
    users.columns = ["user_id", "age"]

    # ----- 1. Basic Statistics -----
    print("\n--- Basic Statistics ---")
    print(f"Total ratings:            {len(ratings):,}")
    print(f"Unique users (ratings):   {ratings.user_id.nunique():,}")
    print(f"Unique books (ratings):   {ratings.isbn.nunique():,}")
    print(f"Total books in DB:        {len(books):,}")
    print(f"Total users in DB:        {len(users):,}")
    density = len(ratings) / (ratings.user_id.nunique() * ratings.isbn.nunique()) * 100
    print(f"Density:                  {density:.6f}%")
    print(f"Sparsity:                 {100 - density:.6f}%")
    print(f"Avg ratings/user:         {ratings.groupby('user_id').size().mean():.1f}")
    print(f"Avg ratings/book:         {ratings.groupby('isbn').size().mean():.1f}")

    # ----- 2. Explicit vs Implicit -----
    explicit = ratings[ratings.rating > 0]
    implicit = ratings[ratings.rating == 0]
    print(f"\n--- Explicit vs Implicit Ratings ---")
    print(f"  Explicit (rating > 0): {len(explicit):,} ({len(explicit)/len(ratings)*100:.1f}%)")
    print(f"  Implicit (rating = 0): {len(implicit):,} ({len(implicit)/len(ratings)*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Explicit (>0)", "Implicit (=0)"], [len(explicit), len(implicit)],
           color=["#3498db", "#95a5a6"], edgecolor="white", width=0.5)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Book-Crossing: Explicit vs Implicit Ratings", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, v in enumerate([len(explicit), len(implicit)]):
        ax.text(i, v + 10000, f"{v:,}", ha="center", fontweight="bold")
    save_fig(fig, os.path.join(output_dir, "explicit_vs_implicit.png"))

    # ----- 3. Rating Distribution (explicit only) -----
    print(f"\n--- Explicit Rating Distribution ---")
    print(f"  Rating range: {explicit.rating.min()} – {explicit.rating.max()}")
    print(f"  Mean rating:  {explicit.rating.mean():.2f}")
    print(f"  Median:       {explicit.rating.median():.1f}")
    rating_counts = explicit.rating.value_counts().sort_index()
    for r, c in rating_counts.items():
        pct = c / len(explicit) * 100
        bar = "█" * int(pct / 2)
        print(f"  {r:>2}: {c:>6,} ({pct:5.1f}%) {bar}")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_bx = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(rating_counts)))
    ax.bar(rating_counts.index, rating_counts.values, color=colors_bx, edgecolor="white", width=0.7)
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Book-Crossing: Explicit Rating Distribution (1–10)", fontsize=14, fontweight="bold")
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    save_fig(fig, os.path.join(output_dir, "rating_distribution.png"))

    # ----- 4. Ratings per User -----
    ratings_per_user = ratings.groupby("user_id").size()
    print(f"\n--- Ratings per User ---")
    print(f"  Min:    {ratings_per_user.min()}")
    print(f"  Max:    {ratings_per_user.max():,}")
    print(f"  Mean:   {ratings_per_user.mean():.1f}")
    print(f"  Median: {ratings_per_user.median():.1f}")
    print(f"  Users with ≥5 ratings: {(ratings_per_user >= 5).sum():,}")

    fig, ax = plt.subplots(figsize=(8, 5))
    # Clip for better visualization (long tail)
    clipped = ratings_per_user.clip(upper=100)
    ax.hist(clipped, bins=50, color="#3498db", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of Ratings (clipped at 100)", fontsize=12)
    ax.set_ylabel("Number of Users", fontsize=12)
    ax.set_title("Book-Crossing: Ratings per User", fontsize=14, fontweight="bold")
    ax.axvline(ratings_per_user.mean(), color="#e74c3c", ls="--",
               label=f"Mean: {ratings_per_user.mean():.0f}")
    ax.axvline(ratings_per_user.median(), color="#2ecc71", ls="--",
               label=f"Median: {ratings_per_user.median():.0f}")
    ax.legend()
    save_fig(fig, os.path.join(output_dir, "ratings_per_user.png"))

    # ----- 5. Ratings per Book -----
    ratings_per_book = ratings.groupby("isbn").size()
    print(f"\n--- Ratings per Book ---")
    print(f"  Min:    {ratings_per_book.min()}")
    print(f"  Max:    {ratings_per_book.max():,}")
    print(f"  Mean:   {ratings_per_book.mean():.1f}")
    print(f"  Median: {ratings_per_book.median():.1f}")
    print(f"  Books with ≥5 ratings: {(ratings_per_book >= 5).sum():,}")

    fig, ax = plt.subplots(figsize=(8, 5))
    clipped_book = ratings_per_book.clip(upper=50)
    ax.hist(clipped_book, bins=50, color="#9b59b6", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of Ratings (clipped at 50)", fontsize=12)
    ax.set_ylabel("Number of Books", fontsize=12)
    ax.set_title("Book-Crossing: Ratings per Book", fontsize=14, fontweight="bold")
    ax.axvline(ratings_per_book.mean(), color="#e74c3c", ls="--",
               label=f"Mean: {ratings_per_book.mean():.0f}")
    ax.legend()
    save_fig(fig, os.path.join(output_dir, "ratings_per_book.png"))

    # ----- 6. Top Authors -----
    author_counts = books["author"].value_counts().head(15)
    print(f"\n--- Top 15 Authors (by # of books) ---")
    for a, c in author_counts.items():
        print(f"  {a:<35s} {c:>4,}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ac_sorted = author_counts.sort_values(ascending=True)
    ax.barh(ac_sorted.index, ac_sorted.values, color="#e67e22", edgecolor="white")
    ax.set_xlabel("Number of Books", fontsize=12)
    ax.set_title("Book-Crossing: Top 15 Authors", fontsize=14, fontweight="bold")
    save_fig(fig, os.path.join(output_dir, "top_authors.png"))

    # ----- 7. Top Publishers -----
    pub_counts = books["publisher"].value_counts().head(15)
    print(f"\n--- Top 15 Publishers ---")
    for p, c in pub_counts.items():
        print(f"  {str(p):<35s} {c:>4,}")

    fig, ax = plt.subplots(figsize=(10, 6))
    pc_sorted = pub_counts.sort_values(ascending=True)
    ax.barh(pc_sorted.index, pc_sorted.values, color="#1abc9c", edgecolor="white")
    ax.set_xlabel("Number of Books", fontsize=12)
    ax.set_title("Book-Crossing: Top 15 Publishers", fontsize=14, fontweight="bold")
    save_fig(fig, os.path.join(output_dir, "top_publishers.png"))

    # ----- 8. User Age Distribution -----
    users_clean = users[pd.to_numeric(users["age"], errors="coerce").notna()].copy()
    users_clean["age"] = users_clean["age"].astype(int)
    users_clean = users_clean[(users_clean.age > 0) & (users_clean.age < 120)]
    print(f"\n--- User Age Distribution ---")
    print(f"  Users with valid age: {len(users_clean):,} / {len(users):,}")
    if len(users_clean) > 0:
        print(f"  Age range: {users_clean.age.min()} – {users_clean.age.max()}")
        print(f"  Mean age:  {users_clean.age.mean():.1f}")
        print(f"  Median:    {users_clean.age.median():.1f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(users_clean.age, bins=range(0, 105, 5), color="#2c3e50", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Number of Users", fontsize=12)
        ax.set_title("Book-Crossing: User Age Distribution", fontsize=14, fontweight="bold")
        ax.axvline(users_clean.age.mean(), color="#e74c3c", ls="--",
                   label=f"Mean: {users_clean.age.mean():.0f}")
        ax.legend()
        save_fig(fig, os.path.join(output_dir, "user_age_distribution.png"))

    # ----- 9. Publication Year -----
    books["year"] = pd.to_numeric(books["year"], errors="coerce")
    books_valid_year = books[(books.year > 1800) & (books.year <= 2025)]
    print(f"\n--- Publication Year ---")
    print(f"  Books with valid year: {len(books_valid_year):,}")
    if len(books_valid_year) > 0:
        print(f"  Year range: {int(books_valid_year.year.min())} – {int(books_valid_year.year.max())}")
        print(f"  Most common decade:")
        books_valid_year = books_valid_year.copy()
        books_valid_year["decade"] = (books_valid_year["year"] // 10 * 10).astype(int)
        decade_counts = books_valid_year["decade"].value_counts().sort_index()
        for d, c in decade_counts.tail(8).items():
            print(f"    {d}s: {c:>6,}")

        fig, ax = plt.subplots(figsize=(10, 5))
        recent = books_valid_year[books_valid_year.year >= 1950]
        year_counts = recent["year"].value_counts().sort_index()
        ax.bar(year_counts.index, year_counts.values, color="#8e44ad", edgecolor="white", width=0.8)
        ax.set_xlabel("Publication Year", fontsize=12)
        ax.set_ylabel("Number of Books", fontsize=12)
        ax.set_title("Book-Crossing: Publication Year (since 1950)", fontsize=14, fontweight="bold")
        save_fig(fig, os.path.join(output_dir, "publication_year.png"))

    # ----- 10. Positive vs Negative for RKGCN -----
    n_pos = len(ratings[ratings.rating > 0])
    n_implicit = len(ratings[ratings.rating == 0])
    print(f"\n--- Implicit Feedback (RKGCN Threshold: rating > 0) ---")
    print(f"  Positive (>0): {n_pos:,} ({n_pos/len(ratings)*100:.1f}%)")
    print(f"  Implicit (=0): {n_implicit:,} ({n_implicit/len(ratings)*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Positive (>0)", "Implicit (=0)"], [n_pos, n_implicit],
           color=["#27ae60", "#e74c3c"], edgecolor="white", width=0.5)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Book-Crossing: Positive vs Implicit for RKGCN", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, v in enumerate([n_pos, n_implicit]):
        ax.text(i, v + 10000, f"{v:,}", ha="center", fontweight="bold")
    save_fig(fig, os.path.join(output_dir, "positive_negative_split.png"))

    print(f"\n  All plots saved to: {output_dir}/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EDA for RKGCN datasets")
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["movie", "book", "both"],
        help="Dataset to analyze: 'movie', 'book', or 'both'",
    )
    parser.add_argument(
        "--output_dir", type=str, default="eda_outputs",
        help="Directory to save plots (default: eda_outputs/)",
    )
    args = parser.parse_args()

    if args.dataset in ("movie", "both"):
        eda_movielens(
            data_dir=os.path.join("datasets", "MovieLens-1M"),
            output_dir=os.path.join(args.output_dir, "MovieLens-1M"),
        )

    if args.dataset in ("book", "both"):
        eda_bookcrossing(
            data_dir=os.path.join("datasets", "Book-Crossing"),
            output_dir=os.path.join(args.output_dir, "Book-Crossing"),
        )

    print("\n" + "=" * 60)
    print("EDA Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
