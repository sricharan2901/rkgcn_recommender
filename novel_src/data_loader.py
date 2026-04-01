"""
data_loader.py — Data loading and ripple/neighbor set construction for RKGCN.

Handles:
  - Loading preprocessed ratings and KG
  - Splitting data into train/validation/test (6:2:2)
  - Constructing user preference (ripple) sets
  - Constructing item neighbor sets for GCN
"""

import os
import numpy as np
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
    Load ratings_final.txt and split into train/val/test (6:2:2).
    Returns numpy arrays of shape (N, 4) with columns [user_id, item_id, label, timestamp_norm].
    """
    ratings_path = os.path.join(data_dir, "ratings_final.txt")
    data = np.loadtxt(ratings_path, dtype=np.float32, delimiter="\t")
    print(f"  Loaded {len(data)} ratings from ratings_final.txt")

    # Normalize timestamps globally to [0, 1]
    timestamps = data[:, 3]
    min_ts = np.min(timestamps)
    max_ts = np.max(timestamps)
    if max_ts > min_ts:
        data[:, 3] = (timestamps - min_ts) / (max_ts - min_ts)
    else:
        data[:, 3] = 1.0

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


def load_kg(data_dir):
    """
    Load kg_final.txt and build adjacency dictionary.
    Returns:
      - kg_dict: {entity_id: [(relation_id, tail_id), ...]}
      - n_entities, n_relations from the data
    """
    kg_path = os.path.join(data_dir, "kg_final.txt")
    triples = np.loadtxt(kg_path, dtype=np.int32, delimiter="\t")
    print(f"  Loaded {len(triples)} KG triples from kg_final.txt")

    kg_dict = defaultdict(list)
    for h, r, t in triples:
        kg_dict[h].append((r, t))
        # Add reverse edges for undirected traversal
        kg_dict[t].append((r, h))

    n_entities = max(triples[:, 0].max(), triples[:, 2].max()) + 1
    n_relations = triples[:, 1].max() + 1

    print(f"  KG entities: {n_entities}, relations: {n_relations}")
    print(f"  Entities with neighbors: {len(kg_dict)}")
    return kg_dict, int(n_entities), int(n_relations)


def construct_preference_sets(train_data, kg_dict, n_hop, n_memory, session_split_ratio=0.2):
    """
    Construct user preference (ripple) sets for each user.
    Splits into long-term and short-term (session) histories.
    """
    print(f"  Constructing dual preference (ripple) sets: {n_hop} hops, {n_memory} memories ...")

    user_interactions = defaultdict(list)
    for i in range(len(train_data)):
        user_id = int(train_data[i, 0])
        item_id = int(train_data[i, 1])
        label = int(train_data[i, 2])
        timestamp = train_data[i, 3] # normalized [0,1]
        if label == 1:
            user_interactions[user_id].append((item_id, timestamp))
            
    user_history_long = defaultdict(list)
    user_history_short = defaultdict(list)
    
    for uid, interactions in user_interactions.items():
        interactions.sort(key=lambda x: x[1]) # chronologically
        n_short = max(1, int(len(interactions) * session_split_ratio))
        if len(interactions) == 1:
            n_short = 1
            n_long = 0
        else:
            n_long = len(interactions) - n_short
            
        user_history_long[uid] = interactions[:n_long]
        user_history_short[uid] = interactions[n_long:]

    n_users = max(int(train_data[:, 0].max()) + 1, len(user_interactions))

    def _build_ripple(history_dict):
        ripple_out = {}
        for user_id in range(n_users):
            ripple_set_user = []
            seed_items = history_dict.get(user_id, [])
            
            seed_dict = {}
            for item_id, ts in seed_items:
                time_delta = 1.0 - ts
                if item_id not in seed_dict or time_delta < seed_dict[item_id]:
                    seed_dict[item_id] = time_delta
                    
            if len(seed_dict) == 0:
                for _ in range(n_hop):
                    ripple_set_user.append((
                        np.zeros(n_memory, dtype=np.int32),
                        np.zeros(n_memory, dtype=np.int32),
                        np.zeros(n_memory, dtype=np.int32),
                        np.zeros(n_memory, dtype=np.float32),
                    ))
                ripple_out[user_id] = ripple_set_user
                continue
                
            for hop in range(n_hop):
                hop_triples = []
                for entity, t_delta in seed_dict.items():
                    if entity in kg_dict:
                        for relation, tail in kg_dict[entity]:
                            hop_triples.append((entity, relation, tail, t_delta))
                            
                if len(hop_triples) == 0:
                    if hop > 0 and len(ripple_set_user) > 0:
                        ripple_set_user.append(ripple_set_user[-1])
                    else:
                        ripple_set_user.append((
                            np.zeros(n_memory, dtype=np.int32),
                            np.zeros(n_memory, dtype=np.int32),
                            np.zeros(n_memory, dtype=np.int32),
                            np.zeros(n_memory, dtype=np.float32),
                        ))
                else:
                    replace = len(hop_triples) < n_memory
                    indices = np.random.choice(len(hop_triples), size=n_memory, replace=replace)
                    heads = np.array([hop_triples[i][0] for i in indices], dtype=np.int32)
                    relations = np.array([hop_triples[i][1] for i in indices], dtype=np.int32)
                    tails = np.array([hop_triples[i][2] for i in indices], dtype=np.int32)
                    deltas = np.array([hop_triples[i][3] for i in indices], dtype=np.float32)
                    ripple_set_user.append((heads, relations, tails, deltas))
                    
                seed_dict = {}
                for h, r, t, d in hop_triples:
                    if t not in seed_dict or d < seed_dict[t]:
                        seed_dict[t] = d
                        
            ripple_out[user_id] = ripple_set_user
        return ripple_out

    ripple_long = _build_ripple(user_history_long)
    ripple_short = _build_ripple(user_history_short)
    
    print(f"  Dual Ripple sets constructed for {n_users} users")
    return ripple_long, ripple_short


def construct_neighbor_sets(kg_dict, n_entities, n_neighbor):
    """
    Construct neighbor sets for each entity in the KG (for GCN entity enhancement).

    For each entity v:
      - N(v) = {(relation, neighbor_entity)} from kg_dict
      - Sampled/padded to size n_neighbor

    Args:
      kg_dict: {entity_id: [(relation_id, tail_id), ...]}
      n_entities: total number of entities
      n_neighbor: neighbor set size N_e

    Returns:
      neighbor_entities: np.array of shape (n_entities, n_neighbor) — neighbor entity IDs
      neighbor_relations: np.array of shape (n_entities, n_neighbor) — neighbor relation IDs
    """
    print(f"  Constructing neighbor sets: {n_neighbor} neighbors per entity ...")

    neighbor_entities = np.zeros((n_entities, n_neighbor), dtype=np.int32)
    neighbor_relations = np.zeros((n_entities, n_neighbor), dtype=np.int32)

    for entity_id in range(n_entities):
        neighbors = kg_dict.get(entity_id, [])
        if len(neighbors) == 0:
            # No neighbors — keep zeros (padding)
            continue

        replace = len(neighbors) < n_neighbor
        indices = np.random.choice(len(neighbors), size=n_neighbor, replace=replace)
        for i, idx in enumerate(indices):
            neighbor_relations[entity_id, i] = neighbors[idx][0]
            neighbor_entities[entity_id, i] = neighbors[idx][1]

    print(f"  Neighbor sets constructed for {n_entities} entities")
    return neighbor_entities, neighbor_relations


def get_batch_data(batch_indices, data, ripple_sets_long, ripple_sets_short, n_hop):
    batch = data[batch_indices]
    users = batch[:, 0].astype(np.int32)
    items = batch[:, 1].astype(np.int32)
    labels = batch[:, 2].astype(np.float32)

    def _extract_memories(ripple_sets):
        memories_h, memories_r, memories_t, memories_time = [], [], [], []
        for hop in range(n_hop):
            h_list, r_list, t_list, time_list = [], [], [], []
            for user_id in users:
                h_list.append(ripple_sets[user_id][hop][0])
                r_list.append(ripple_sets[user_id][hop][1])
                t_list.append(ripple_sets[user_id][hop][2])
                time_list.append(ripple_sets[user_id][hop][3])
            memories_h.append(np.array(h_list, dtype=np.int32))
            memories_r.append(np.array(r_list, dtype=np.int32))
            memories_t.append(np.array(t_list, dtype=np.int32))
            memories_time.append(np.array(time_list, dtype=np.float32))
        return memories_h, memories_r, memories_t, memories_time

    h_long, r_long, t_long, time_long = _extract_memories(ripple_sets_long)
    h_short, r_short, t_short, time_short = _extract_memories(ripple_sets_short)

    return {
        "users": users,
        "items": items,
        "labels": labels,
        "memories_h_long": h_long,
        "memories_r_long": r_long,
        "memories_t_long": t_long,
        "memories_time_long": time_long,
        "memories_h_short": h_short,
        "memories_r_short": r_short,
        "memories_t_short": t_short,
        "memories_time_short": time_short,
    }


def load_data(data_dir, n_hop, n_memory, n_neighbor):
    """
    Main data loading function. Returns all data structures needed for RKGCN.

    Args:
      data_dir: path to preprocessed data directory
      n_hop: number of preference propagation hops
      n_memory: preference set size per hop
      n_neighbor: neighbor set size for GCN

    Returns dict with:
      - train_data, val_data, test_data: numpy arrays (N, 3)
      - n_users, n_items, n_entities, n_relations
      - ripple_sets: preference sets per user
      - neighbor_entities, neighbor_relations: neighbor sets per entity
    """
    print(f"\n{'='*60}")
    print(f"Loading data from {data_dir}")
    print(f"{'='*60}")

    # Load ratings
    train_data, val_data, test_data = load_ratings(data_dir)

    # Load KG
    kg_dict, n_entities, n_relations = load_kg(data_dir)

    # Load metadata
    meta = load_metadata(data_dir)
    n_users = meta["n_users"]
    n_items = meta["n_items"]

    # Construct preference (ripple) sets
    np.random.seed(42)
    ripple_long, ripple_short = construct_preference_sets(train_data, kg_dict, n_hop, n_memory)

    # Construct neighbor sets for GCN
    np.random.seed(42)
    neighbor_entities, neighbor_relations = construct_neighbor_sets(
        kg_dict, n_entities, n_neighbor
    )

    print(f"\nData loading complete:")
    print(f"  Users: {n_users}, Items: {n_items}")
    print(f"  Entities: {n_entities}, Relations: {n_relations}")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print()

    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "n_users": n_users,
        "n_items": n_items,
        "n_entities": n_entities,
        "n_relations": n_relations,
        "ripple_long": ripple_long,
        "ripple_short": ripple_short,
        "neighbor_entities": neighbor_entities,
        "neighbor_relations": neighbor_relations,
        "kg_dict": kg_dict,
    }
