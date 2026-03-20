"""
preprocess.py — Data preprocessing for RKGCN.

Handles:
  - MovieLens-1M: Parse ratings.dat + movies.dat, convert to implicit feedback,
    build KG from movie genres
  - Book-Crossing: Parse Ratings.csv + Books.csv, convert to implicit feedback,
    build KG from book metadata (author, publisher, year)
  - Re-index all IDs to contiguous integers
  - Save ratings_final.txt, kg_final.txt, metadata.txt

Usage:
  python src/preprocess.py --dataset movie --data_dir datasets/MovieLens-1M
  python src/preprocess.py --dataset book --data_dir datasets/Book-Crossing
"""

import argparse
import os
import numpy as np
import pandas as pd
from collections import defaultdict


# ============================================================
# MovieLens-1M
# ============================================================

def preprocess_movie(data_dir):
    """
    Preprocess MovieLens-1M dataset.
    - Reads ratings.dat (UserID::MovieID::Rating::Timestamp)
    - Converts to implicit feedback: rating >= 4 -> 1, sample negatives -> 0
    """
    ratings_file = os.path.join(data_dir, "ratings.dat")
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(
            f"ratings.dat not found in {data_dir}. "
            f"Download MovieLens-1M from https://grouplens.org/datasets/movielens/1m/"
        )

    print("[MovieLens-1M] Loading ratings.dat ...")
    ratings = pd.read_csv(
        ratings_file, sep="::", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
    )
    print(f"  Total ratings: {len(ratings)}")

    # Positive: rating >= 4
    positive = ratings[ratings["rating"] >= 4][["user_id", "movie_id"]].copy()
    positive["label"] = 1
    positive.rename(columns={"movie_id": "item_id"}, inplace=True)
    print(f"  Positive interactions (rating >= 4): {len(positive)}")

    # Sample negatives (1:1 ratio)
    all_items = set(ratings["movie_id"].unique())
    user_pos = positive.groupby("user_id")["item_id"].apply(set).to_dict()

    neg_rows = []
    np.random.seed(42)
    all_items_array = np.array(list(all_items))
    for uid, pos_items in user_pos.items():
        n_neg = min(len(pos_items), len(all_items_array) - len(pos_items))
        if n_neg == 0:
            continue
        
        sampled = set()
        while len(sampled) < n_neg:
            cands = np.random.choice(all_items_array, size=n_neg * 2, replace=True)
            for cand in cands:
                if cand not in pos_items:
                    sampled.add(cand)
                    if len(sampled) == n_neg:
                        break
                        
        for iid in sampled:
            neg_rows.append({"user_id": uid, "item_id": iid, "label": 0})

    negative = pd.DataFrame(neg_rows)
    ratings_final = pd.concat([positive, negative], ignore_index=True)
    ratings_final = ratings_final.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Total (pos + neg): {len(ratings_final)}")

    return ratings_final


def build_movie_kg(data_dir):
    """
    Build knowledge graph from movies.dat (MovieID::Title::Genres).
    Creates triples: (movie_entity, has_genre, genre_entity)
    """
    movies_file = os.path.join(data_dir, "movies.dat")
    if not os.path.exists(movies_file):
        raise FileNotFoundError(f"movies.dat not found in {data_dir}")

    print("[MovieLens-1M] Building KG from movies.dat (genres) ...")
    movies = pd.read_csv(
        movies_file, sep="::", header=None,
        names=["movie_id", "title", "genres"],
        engine="python", encoding="latin-1",
    )

    triples = []  # (head, relation, tail) as strings
    for _, row in movies.iterrows():
        movie_entity = f"movie_{row['movie_id']}"
        genres = row["genres"].split("|")
        for genre in genres:
            genre_entity = f"genre_{genre.strip()}"
            triples.append((movie_entity, "has_genre", genre_entity))

    # Also extract year from title if possible (e.g., "Toy Story (1995)")
    for _, row in movies.iterrows():
        movie_entity = f"movie_{row['movie_id']}"
        title = str(row["title"])
        if "(" in title and ")" in title:
            year_str = title[title.rfind("(") + 1 : title.rfind(")")]
            if year_str.isdigit():
                year_entity = f"year_{year_str}"
                triples.append((movie_entity, "released_in", year_entity))

    print(f"  KG triples from genres + years: {len(triples)}")

    # Build item-to-entity mapping (movie_id -> entity name)
    item2entity = {}
    for _, row in movies.iterrows():
        item2entity[str(row["movie_id"])] = f"movie_{row['movie_id']}"

    return triples, item2entity


# ============================================================
# Book-Crossing
# ============================================================

def preprocess_book(data_dir):
    """
    Preprocess Book-Crossing dataset.
    - Reads Ratings.csv (User-ID;ISBN;Rating)
    - Converts to implicit feedback: rating > 0 -> 1, sample negatives -> 0
    """
    ratings_file = os.path.join(data_dir, "Ratings.csv")
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(
            f"Ratings.csv not found in {data_dir}. "
            f"Download Book-Crossing dataset from Kaggle."
        )

    print("[Book-Crossing] Loading Ratings.csv ...")
    ratings = pd.read_csv(
        ratings_file, sep=";", encoding="latin-1", on_bad_lines="skip",
    )
    ratings.columns = ["user_id", "isbn", "rating"]
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings.dropna(subset=["rating"], inplace=True)
    ratings["rating"] = ratings["rating"].astype(int)
    print(f"  Total ratings: {len(ratings)}")

    # Positive: any rating > 0
    positive = ratings[ratings["rating"] > 0][["user_id", "isbn"]].copy()
    positive["label"] = 1
    positive.rename(columns={"isbn": "item_id"}, inplace=True)
    print(f"  Positive interactions (rating > 0): {len(positive)}")

    # Sample negatives (1:1 ratio)
    all_items = set(ratings["isbn"].unique())
    user_pos = positive.groupby("user_id")["item_id"].apply(set).to_dict()

    neg_rows = []
    np.random.seed(42)
    all_items_array = np.array(list(all_items))
    for uid, pos_items in user_pos.items():
        n_neg = min(len(pos_items), len(all_items_array) - len(pos_items))
        if n_neg == 0:
            continue
        
        sampled = set()
        while len(sampled) < n_neg:
            cands = np.random.choice(all_items_array, size=n_neg * 2, replace=True)
            for cand in cands:
                if cand not in pos_items:
                    sampled.add(cand)
                    if len(sampled) == n_neg:
                        break
                        
        for iid in sampled:
            neg_rows.append({"user_id": uid, "item_id": iid, "label": 0})

    negative = pd.DataFrame(neg_rows)
    ratings_final = pd.concat([positive, negative], ignore_index=True)
    ratings_final = ratings_final.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Total (pos + neg): {len(ratings_final)}")

    return ratings_final


def build_book_kg(data_dir):
    """
    Build knowledge graph from Books.csv (ISBN;Title;Author;Year;Publisher).
    Creates triples:
      - (book_entity, written_by, author_entity)
      - (book_entity, published_by, publisher_entity)
      - (book_entity, published_in, year_entity)
    """
    books_file = os.path.join(data_dir, "Books.csv")
    if not os.path.exists(books_file):
        raise FileNotFoundError(f"Books.csv not found in {data_dir}")

    print("[Book-Crossing] Building KG from Books.csv (author, publisher, year) ...")
    books = pd.read_csv(
        books_file, sep=";", encoding="latin-1", on_bad_lines="skip",
        low_memory=False,
    )
    # Standardize columns
    books.columns = ["isbn", "title", "author", "year", "publisher"]

    triples = []
    item2entity = {}

    for _, row in books.iterrows():
        isbn = str(row["isbn"]).strip()
        book_entity = f"book_{isbn}"
        item2entity[isbn] = book_entity

        # Author
        author = str(row.get("author", "")).strip()
        if author and author.lower() != "nan":
            author_entity = f"author_{author.replace(' ', '_')}"
            triples.append((book_entity, "written_by", author_entity))

        # Publisher
        publisher = str(row.get("publisher", "")).strip()
        if publisher and publisher.lower() != "nan":
            publisher_entity = f"publisher_{publisher.replace(' ', '_')}"
            triples.append((book_entity, "published_by", publisher_entity))

        # Year
        year = str(row.get("year", "")).strip()
        if year.isdigit() and int(year) > 0:
            year_entity = f"year_{year}"
            triples.append((book_entity, "published_in", year_entity))

    print(f"  KG triples from book metadata: {len(triples)}")
    return triples, item2entity


# ============================================================
# Common: Reindex and Save
# ============================================================

def reindex_and_save(ratings_final, kg_triples, item2entity, data_dir):
    """
    Re-index all user, item, entity, and relation IDs to contiguous integers.
    Save ratings_final.txt, kg_final.txt, metadata.txt.
    """
    # --- Build entity and relation vocabularies ---
    entity_set = set()
    relation_set = set()
    for h, r, t in kg_triples:
        entity_set.update([h, t])
        relation_set.add(r)

    # Include all item entities
    for ent in item2entity.values():
        entity_set.add(ent)

    entity_list = sorted(entity_set)
    entity2idx = {e: idx for idx, e in enumerate(entity_list)}

    relation_list = sorted(relation_set)
    relation2idx = {r: idx for idx, r in enumerate(relation_list)}

    # --- Build item_id -> entity_idx mapping ---
    item_id_to_entity_idx = {}
    for item_id_str, entity_name in item2entity.items():
        if entity_name in entity2idx:
            item_id_to_entity_idx[item_id_str] = entity2idx[entity_name]

    # --- Filter ratings to items that have KG entities ---
    before = len(ratings_final)
    ratings_final = ratings_final[
        ratings_final["item_id"].astype(str).isin(item_id_to_entity_idx.keys())
    ].copy()
    print(f"  Ratings after filtering to KG-mapped items: {len(ratings_final)} (removed {before - len(ratings_final)})")

    if len(ratings_final) == 0:
        raise ValueError("No ratings left after filtering! Check that item IDs match between ratings and KG.")

    # --- Re-index users ---
    unique_users = sorted(ratings_final["user_id"].unique(), key=str)
    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    ratings_final["user_id"] = ratings_final["user_id"].map(user2idx)

    # --- Re-index items to entity indices ---
    ratings_final["item_id"] = ratings_final["item_id"].astype(str).map(item_id_to_entity_idx)

    # Drop any rows where item mapping failed
    ratings_final.dropna(subset=["item_id"], inplace=True)
    ratings_final["item_id"] = ratings_final["item_id"].astype(int)

    # --- Re-index KG ---
    kg_reindexed = []
    for h, r, t in kg_triples:
        if h in entity2idx and t in entity2idx and r in relation2idx:
            kg_reindexed.append((entity2idx[h], relation2idx[r], entity2idx[t]))

    # --- Save ratings_final.txt ---
    ratings_path = os.path.join(data_dir, "ratings_final.txt")
    ratings_final[["user_id", "item_id", "label"]].to_csv(
        ratings_path, sep="\t", index=False, header=False,
    )
    print(f"  Saved {ratings_path} ({len(ratings_final)} entries)")

    # --- Save kg_final.txt ---
    kg_path = os.path.join(data_dir, "kg_final.txt")
    with open(kg_path, "w") as f:
        for h, r, t in kg_reindexed:
            f.write(f"{h}\t{r}\t{t}\n")
    print(f"  Saved {kg_path} ({len(kg_reindexed)} triples)")

    # --- Save metadata.txt ---
    n_entities = len(entity_list)
    n_relations = len(relation_list)
    n_users = len(unique_users)
    n_items = ratings_final["item_id"].nunique()

    meta_path = os.path.join(data_dir, "metadata.txt")
    with open(meta_path, "w") as f:
        f.write(f"n_users\t{n_users}\n")
        f.write(f"n_items\t{n_items}\n")
        f.write(f"n_entities\t{n_entities}\n")
        f.write(f"n_relations\t{n_relations}\n")
        f.write(f"n_ratings\t{len(ratings_final)}\n")
        f.write(f"n_kg_triples\t{len(kg_reindexed)}\n")
    print(f"  Saved metadata to {meta_path}")

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_entities": n_entities,
        "n_relations": n_relations,
    }


# ============================================================
# Main Pipeline
# ============================================================

def preprocess(dataset, data_dir):
    """Main preprocessing pipeline."""
    print(f"\n{'='*60}")
    print(f"Preprocessing dataset: {dataset}")
    print(f"Data directory: {data_dir}")
    print(f"{'='*60}\n")

    # Step 1: Preprocess ratings
    if dataset == "movie":
        ratings_final = preprocess_movie(data_dir)
        kg_triples, item2entity = build_movie_kg(data_dir)
    elif dataset == "book":
        ratings_final = preprocess_book(data_dir)
        kg_triples, item2entity = build_book_kg(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'movie' or 'book'.")

    # Step 2: Re-index and save
    meta = reindex_and_save(ratings_final, kg_triples, item2entity, data_dir)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"  Users:     {meta['n_users']}")
    print(f"  Items:     {meta['n_items']}")
    print(f"  Entities:  {meta['n_entities']}")
    print(f"  Relations: {meta['n_relations']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for RKGCN")
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["movie", "book"],
        help="Dataset to preprocess: 'movie' (MovieLens-1M) or 'book' (Book-Crossing)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to data directory.",
    )
    args = parser.parse_args()

    if args.data_dir is None:
        if args.dataset == "movie":
            args.data_dir = os.path.join("datasets", "MovieLens-1M")
        else:
            args.data_dir = os.path.join("datasets", "Book-Crossing")

    preprocess(args.dataset, args.data_dir)
