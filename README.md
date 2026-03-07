# RKGCN — Ripple Knowledge Graph Convolutional Networks

Implementation of the **RKGCN** model for recommendation systems, based on the paper:

> **"Ripple Knowledge Graph Convolutional Networks for Recommendation Systems"**
> Chen Li et al., *Machine Intelligence Research*, 2024

RKGCN combines two complementary approaches for knowledge-graph-enhanced recommendations:
1. **User Preference Aggregation** (inspired by RippleNet) — propagates user preferences through multi-hop KG links
2. **Entity Enhancement via GCN** (inspired by KGCN) — enriches item embeddings using graph convolution over KG neighborhoods

---

## Project Structure

```
IT414 Project/
├── README.md                  # This file
│
├── src/                       # All source code
│   ├── __init__.py
│   ├── main.py                # CLI runner — preprocessing, training, evaluation
│   ├── eda.py                 # Exploratory Data Analysis for both datasets
│   ├── requirements.txt       # Python dependencies
│   ├── preprocess.py          # Data preprocessing + KG construction from metadata
│   ├── data_loader.py         # Data loading, train/val/test split, ripple & neighbor sets
│   ├── model.py               # RKGCN model (TensorFlow/Keras)
│   ├── train.py               # Training loop with validation tracking
│   └── evaluate.py            # AUC and Accuracy evaluation
│
├── datasets/                  # Raw datasets
│   ├── MovieLens-1M/          # ratings.dat, movies.dat, users.dat
│   └── Book-Crossing/         # Ratings.csv, Books.csv, Users.csv
│
├── eda_outputs/               # Generated EDA plots (after running eda.py)
│   ├── MovieLens-1M/
│   └── Book-Crossing/
│
└── DWMenv/                    # Python virtual environment
```

---

## Requirements

| Package | Version | Purpose |
|---|---|---|
| TensorFlow | ≥ 2.10 | Deep learning framework |
| NumPy | ≥ 1.21 | Numerical operations |
| Pandas | ≥ 1.3 | Data loading & manipulation |
| Scikit-learn | ≥ 1.0 | Evaluation metrics (AUC, Accuracy) |
| Matplotlib | ≥ 3.5 | EDA plots & visualizations |

All listed in `src/requirements.txt`.

---

## Virtual Environment

The project uses a Python virtual environment named **`DWMenv`**.

### Setup
```bash
# Create the venv (already done)
python3 -m venv DWMenv

# Activate it
source DWMenv/bin/activate

# Install dependencies
pip install -r src/requirements.txt
```

### Activating (every session)
```bash
source DWMenv/bin/activate
```

---

## Datasets

### MovieLens-1M
- **Source**: [GroupLens](https://grouplens.org/datasets/movielens/1m/)
- **Size**: 1,000,209 ratings from 6,040 users on 3,706 movies
- **Rating scale**: 1–5 stars
- **RKGCN threshold**: rating ≥ 4 → positive interaction
- **KG**: Built from movie genres (18 genres) and release years

### Book-Crossing
- **Source**: [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)
- **Size**: 1,149,780 ratings from 105,283 users on 340,556 books
- **Rating scale**: 0–10 (0 = implicit interaction)
- **RKGCN threshold**: rating > 0 → positive interaction
- **KG**: Built from book metadata (author, publisher, year)

---

## Usage

### Step 1: Activate the virtual environment
```bash
source DWMenv/bin/activate
```

### Step 2: Run EDA (optional — generates statistics and plots)
```bash
python -m src.eda --dataset both
```

### Step 3: Preprocess data
```bash
python -m src.main --dataset movie --preprocess
python -m src.main --dataset book --preprocess
```

### Step 4: Train and evaluate
```bash
# MovieLens-1M (paper defaults)
python -m src.main --dataset movie --dim 8 --n_hop 2 --n_memory 32 --lr 0.02 --n_epoch 10

# Book-Crossing (paper defaults)
python -m src.main --dataset book --dim 4 --n_hop 1 --n_memory 16 --lr 0.01 --n_epoch 10

# Multi-run averaging
python -m src.main --dataset movie --n_runs 5
```

### Hyperparameters (CLI arguments)

| Argument | Default | Description |
|---|---|---|
| `--dataset` | *required* | `movie` or `book` |
| `--dim` | 8 | Embedding dimension |
| `--n_hop` | 2 | Number of preference propagation hops |
| `--n_memory` | 32 | Preference set size per hop |
| `--n_neighbor` | 16 | Neighbor set size for GCN |
| `--kge_weight` | 0.01 | KG embedding loss weight |
| `--l2` | 1e-7 | L2 regularization weight |
| `--lr` | 0.02 | Learning rate |
| `--batch_size` | 1024 | Training batch size |
| `--n_epoch` | 10 | Number of training epochs |
| `--gcn_iter` | 1 | GCN aggregation iterations |
| `--n_runs` | 1 | Number of runs to average |

---

## Exploratory Data Analysis

### MovieLens-1M

#### Basic Statistics

| Metric | Value |
|---|---|
| Total ratings | 1,000,209 |
| Unique users | 6,040 |
| Unique movies | 3,706 |
| Total movies in DB | 3,883 |
| Density | 4.47% |
| **Sparsity** | **95.53%** |
| Avg ratings/user | 165.6 |
| Avg ratings/movie | 269.9 |
| Mean rating | 3.58 |
| Median rating | 4.0 |

#### Rating Distribution

| Rating | Count | Percentage |
|---|---|---|
| ⭐ 1 | 56,174 | 5.6% |
| ⭐ 2 | 107,557 | 10.8% |
| ⭐ 3 | 261,197 | 26.1% |
| ⭐ 4 | 348,971 | 34.9% |
| ⭐ 5 | 226,310 | 22.6% |

The distribution is **left-skewed** — users tend to rate higher. The 4-star rating is the most common.

![Rating Distribution](eda_outputs/MovieLens-1M/rating_distribution.png)

#### Ratings per User

| Metric | Value |
|---|---|
| Min | 20 |
| Max | 2,314 |
| Mean | 165.6 |
| Median | 96.0 |
| Std Dev | 192.7 |

A **minimum of 20 ratings per user** indicates this is a well-filtered dataset. However, the high std dev (192.7) shows a **long-tail distribution** — most users rate a moderate number of movies while some power users rate thousands.

![Ratings per User](eda_outputs/MovieLens-1M/ratings_per_user.png)

#### Ratings per Movie

| Metric | Value |
|---|---|
| Min | 1 |
| Max | 3,428 |
| Mean | 269.9 |
| Median | 123.5 |

Movies also follow a **long-tail distribution**. Popular movies like "American Beauty" and "Star Wars" receive thousands of ratings while niche films may have only 1.

![Ratings per Movie](eda_outputs/MovieLens-1M/ratings_per_movie.png)

#### Genre Distribution

18 unique genres in the dataset:

| Genre | Movies | Genre | Movies |
|---|---|---|---|
| Drama | 1,603 | Crime | 211 |
| Comedy | 1,200 | War | 143 |
| Action | 503 | Documentary | 127 |
| Thriller | 492 | Musical | 114 |
| Romance | 471 | Mystery | 106 |
| Horror | 343 | Animation | 105 |
| Adventure | 283 | Fantasy | 68 |
| Sci-Fi | 276 | Western | 68 |
| Children's | 251 | Film-Noir | 44 |

**Drama** and **Comedy** dominate, together accounting for nearly half of all genre tags.

![Genre Distribution](eda_outputs/MovieLens-1M/genre_distribution.png)

#### User Demographics

| Gender | Count | Percentage |
|---|---|---|
| Male | 4,331 | 71.7% |
| Female | 1,709 | 28.3% |

| Age Group | Count |
|---|---|
| Under 18 | 222 |
| 18–24 | 1,103 |
| 25–34 | 2,096 |
| 35–44 | 1,193 |
| 45–49 | 550 |
| 50–55 | 496 |
| 56+ | 380 |

The user base skews **male** (71.7%) and **young adult** (25–34 is the largest group).

![User Demographics](eda_outputs/MovieLens-1M/user_demographics.png)

#### Temporal Analysis

Ratings span from **April 2000** to **February 2003**, with the vast majority collected in the year 2000:

| Year | Ratings |
|---|---|
| 2000 | 904,757 |
| 2001 | 68,058 |
| 2002 | 24,046 |
| 2003 | 3,348 |

![Temporal Analysis](eda_outputs/MovieLens-1M/temporal_analysis.png)

#### Implicit Feedback Split (RKGCN)

For RKGCN, ratings ≥ 4 are treated as **positive** and the rest as negative:

| Class | Count | Percentage |
|---|---|---|
| Positive (≥ 4) | 575,281 | 57.5% |
| Negative (< 4) | 424,928 | 42.5% |

A **57.5%/42.5% split** — reasonably balanced for binary classification.

![Positive vs Negative Split](eda_outputs/MovieLens-1M/positive_negative_split.png)

---

## Model Architecture

```
User ──► Ripple Set (H hops) ──► Attention + Aggregation ──► User Embedding (o_u)
                                                                    │
Item ──► KG Neighbors ──► GCN (user-aware weights) ──► Enhanced Item Embedding (e_v')
                                                                    │
                                                    Prediction: σ(o_u · e_v')
```

**Loss Function**: L = L_rec (cross-entropy) + λ₁ · L_KGE + λ₂ · L_reg

---

## References

1. Chen Li et al., "Ripple Knowledge Graph Convolutional Networks for Recommendation Systems", *Machine Intelligence Research*, 2024.
2. Wang et al., "RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems", *CIKM*, 2018.
3. Wang et al., "Knowledge Graph Convolutional Networks for Recommender Systems", *WWW*, 2019.
