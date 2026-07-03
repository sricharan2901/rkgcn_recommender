import os
import bisect
import numpy as np
import tensorflow as tf
from collections import defaultdict


def load_metadata(data_dir):
    """Load dataset metadata (counts of users, items, entities, relations)."""
    meta_path = os.path.join(data_dir, "metadata.txt")
    meta = {}
    with open(meta_path, "r") as f:
        for line in f:
            key, value = line.strip().split("\t")
            meta[key] = int(value)
    return meta


def load_ratings(data_dir):
    """
    Load ratings_final.txt and split into train/val/test (6:2:2) using seed 42.
    Returns:
      train_data, val_data, test_data: arrays with columns [user_id, item_id, label, timestamp]
    """
    ratings_path = os.path.join(data_dir, "ratings_final.txt")
    data = np.loadtxt(ratings_path, dtype=np.float32, delimiter="\t")
    print(f"  Loaded {len(data)} ratings from ratings_final.txt")

    # Shuffle
    np.random.seed(42)
    np.random.shuffle(data)

    # Split 6:2:2
    n = len(data)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    test_data = data[n_train + n_val :]

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data


def construct_adj_matrix(train_data, n_users, n_entities):
    """
    Construct sparse normalized adjacency matrix for LightGCN/NGCF.
    A = [0, R; R^T, 0]
    A_norm = D^-0.5 * A * D^-0.5
    """
    print("  Constructing GCN normalized bipartite adjacency matrix ...")
    
    # Filter to positive interactions in training data
    pos_train = train_data[train_data[:, 2] == 1]
    
    user_nodes = pos_train[:, 0].astype(np.int32)
    item_nodes = pos_train[:, 1].astype(np.int32)

    # Calculate degrees
    user_deg = np.zeros(n_users)
    item_deg = np.zeros(n_entities)

    for u, i in zip(user_nodes, item_nodes):
        user_deg[u] += 1
        item_deg[i] += 1

    # Edge indices
    # We map user u to index u, and item i to index n_users + i
    indices = []
    values = []

    for u, i in zip(user_nodes, item_nodes):
        deg_u = user_deg[u]
        deg_i = item_deg[i]
        
        if deg_u > 0 and deg_i > 0:
            norm_val = 1.0 / np.sqrt(deg_u * deg_i)
            
            # User to Item
            indices.append([u, n_users + i])
            values.append(norm_val)
            
            # Item to User (symmetric matrix)
            indices.append([n_users + i, u])
            values.append(norm_val)

    # Convert to SparseTensor
    indices = np.array(indices, dtype=np.int64)
    values = np.array(values, dtype=np.float32)
    
    # Ensure no duplicates (if multiple ratings for same pair exist, though preprocess.py filters them)
    # If indices list is empty (cold-start/no positive ratings), return empty sparse matrix
    if len(indices) == 0:
        indices = np.zeros((0, 2), dtype=np.int64)
        values = np.zeros((0,), dtype=np.float32)
        
    n_nodes = n_users + n_entities
    
    # Sort indices for TensorFlow sparse tensor requirements
    sort_idx = np.lexsort((indices[:, 1], indices[:, 0]))
    indices = indices[sort_idx]
    values = values[sort_idx]

    adj_matrix = tf.sparse.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[n_nodes, n_nodes]
    )
    
    print(f"  Bipartite graph created with {n_nodes} nodes and {len(indices)} directed edges.")
    return adj_matrix


def construct_sequence_history(train_data):
    """
    Build positive item history mappings for each user in the training set.
    """
    print("  Constructing user chronological sequence histories from train set ...")
    user_pos_items = defaultdict(list)
    user_pos_times = defaultdict(list)

    # Positive interactions in training set
    pos_train = train_data[train_data[:, 2] == 1]
    
    for row in pos_train:
        u = int(row[0])
        i = int(row[1])
        t = row[3]  # timestamp
        user_pos_items[u].append(i)
        user_pos_times[u].append(t)

    # Sort each user's history chronologically by timestamp
    for u in list(user_pos_items.keys()):
        items = user_pos_items[u]
        times = user_pos_times[u]
        
        # Sort items based on timestamps
        sorted_pairs = sorted(zip(items, times), key=lambda x: x[1])
        user_pos_items[u] = [pair[0] for pair in sorted_pairs]
        user_pos_times[u] = [pair[1] for pair in sorted_pairs]

    return user_pos_items, user_pos_times


def get_seq_history_batch(users, target_times, target_items, user_pos_items, user_pos_times, max_len, n_items):
    """
    Generate historical sequence for a batch of interactions.
    Applies strict temporal filtering to prevent data leakage.
    For Book-Crossing (where target_times are all 0), uses train history excluding the target item itself.
    """
    batch_seq = []
    for u, t_target, i_target in zip(users, target_times, target_items):
        history = []
        pos_items = user_pos_items.get(u, [])
        pos_times = user_pos_times.get(u, [])
        
        for item, t in zip(pos_items, pos_times):
            if t_target > 0.0:
                # Temporal filtering (only items strictly prior to current target interaction)
                if t < t_target:
                    history.append(item)
            else:
                # Book-Crossing: use train history, excluding the target item itself to prevent self-matching leakage
                if item != i_target:
                    history.append(item)
                    
        # Pad and truncate
        if len(history) > max_len:
            history = history[-max_len:]
        else:
            # Padding index is n_items (out-of-bounds index, safe for Embedding layers of size n_items+1 or n_items+2)
            history = [n_items] * (max_len - len(history)) + history
            
        batch_seq.append(history)
        
    return np.array(batch_seq, dtype=np.int32)


def precompute_history_sequences(data_split, user_pos_items, user_pos_times, max_len, n_entities):
    """
    Precompute the historical sequences for each sample in a data split.
    This avoids slow per-batch Python loops during training.
    """
    print(f"  Precomputing historical sequences for split of size {len(data_split)} ...")
    users = data_split[:, 0].astype(np.int32)
    items = data_split[:, 1].astype(np.int32)
    times = data_split[:, 3]
    
    seqs = []
    for u, t_target, i_target in zip(users, times, items):
        if t_target > 0.0:
            # MovieLens-1M: find all clicked items where click_time < t_target
            pos_times = user_pos_times.get(u, [])
            pos_items = user_pos_items.get(u, [])
            idx = bisect.bisect_left(pos_times, t_target)
            history = pos_items[:idx]
        else:
            # Book-Crossing: all items in train_data except the current target item
            pos_items = user_pos_items.get(u, [])
            history = [x for x in pos_items if x != i_target]
            
        if len(history) > max_len:
            history = history[-max_len:]
        else:
            history = [n_entities] * (max_len - len(history)) + history
            
        seqs.append(history)
        
    return np.array(seqs, dtype=np.int32)


def load_baseline_data(data_dir):
    """
    Main loader for baseline models.
    """
    print(f"Loading data for baselines from {data_dir}...")
    
    # Load metadata
    meta = load_metadata(data_dir)
    n_users = meta["n_users"]
    n_items = meta["n_items"]
    n_entities = meta["n_entities"]
    
    # Load ratings and split
    train_data, val_data, test_data = load_ratings(data_dir)
    
    # Construct representations
    adj_matrix = construct_adj_matrix(train_data, n_users, n_entities)
    user_pos_items, user_pos_times = construct_sequence_history(train_data)
    
    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "n_users": n_users,
        "n_items": n_items,
        "n_entities": n_entities,
        "adj_matrix": adj_matrix,
        "user_pos_items": user_pos_items,
        "user_pos_times": user_pos_times
    }
