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
    Returns numpy arrays of shape (N, 3) with columns [user_id, item_id, label].
    """
    ratings_path = os.path.join(data_dir, "ratings_final.txt")
    data = np.loadtxt(ratings_path, dtype=np.int32, delimiter="\t")
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


def construct_preference_sets(train_data, kg_dict, n_hop, n_memory):
    """
    Construct user preference (ripple) sets for each user.

    For each user u:
      - Hop 0: items the user interacted with (positive only)
      - Hop k: KG triples whose head entity is in the tail set of hop k-1
      - Each hop's set is sampled/padded to size n_memory

    Args:
      train_data: (N, 3) array [user_id, item_id, label]
      kg_dict: {entity_id: [(relation_id, tail_id), ...]}
      n_hop: number of hops H
      n_memory: ripple set size per hop N_p

    Returns:
      ripple_sets: dict {user_id: [(heads_k, relations_k, tails_k) for k in range(n_hop)]}
        where each is a numpy array of shape (n_memory,)
    """
    print(f"  Constructing preference (ripple) sets: {n_hop} hops, {n_memory} memories ...")

    # Get positive items per user
    user_history = defaultdict(set)
    for user_id, item_id, label in train_data:
        if label == 1:
            user_history[user_id].add(item_id)

    ripple_sets = {}
    n_users = max(train_data[:, 0]) + 1

    for user_id in range(n_users):
        ripple_set_user = []
        # Start with user's clicked items as seed entities
        seed_entities = user_history.get(user_id, set())

        if len(seed_entities) == 0:
            # Handle cold-start: use zeros as padding
            for _ in range(n_hop):
                ripple_set_user.append((
                    np.zeros(n_memory, dtype=np.int32),
                    np.zeros(n_memory, dtype=np.int32),
                    np.zeros(n_memory, dtype=np.int32),
                ))
            ripple_sets[user_id] = ripple_set_user
            continue

        for hop in range(n_hop):
            # Collect all (head, relation, tail) triples from seed entities
            hop_triples = []
            for entity in seed_entities:
                if entity in kg_dict:
                    for relation, tail in kg_dict[entity]:
                        hop_triples.append((entity, relation, tail))

            if len(hop_triples) == 0:
                # No outgoing edges — repeat previous hop or pad with zeros
                if hop > 0 and len(ripple_set_user) > 0:
                    ripple_set_user.append(ripple_set_user[-1])
                else:
                    ripple_set_user.append((
                        np.zeros(n_memory, dtype=np.int32),
                        np.zeros(n_memory, dtype=np.int32),
                        np.zeros(n_memory, dtype=np.int32),
                    ))
            else:
                # Sample or pad to n_memory
                replace = len(hop_triples) < n_memory
                indices = np.random.choice(
                    len(hop_triples), size=n_memory, replace=replace
                )
                heads = np.array([hop_triples[i][0] for i in indices], dtype=np.int32)
                relations = np.array([hop_triples[i][1] for i in indices], dtype=np.int32)
                tails = np.array([hop_triples[i][2] for i in indices], dtype=np.int32)
                ripple_set_user.append((heads, relations, tails))

            # Update seed entities for next hop
            seed_entities = set()
            for _, _, t in hop_triples:
                seed_entities.add(t)

        ripple_sets[user_id] = ripple_set_user

    print(f"  Ripple sets constructed for {len(ripple_sets)} users")
    return ripple_sets


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


def get_batch_data(batch_indices, data, ripple_sets, n_hop):
    """
    Prepare a mini-batch of data for training/evaluation.

    Args:
      batch_indices: indices into data array
      data: (N, 3) array [user_id, item_id, label]
      ripple_sets: dict from construct_preference_sets
      n_hop: number of hops

    Returns dict with:
      - users: (batch_size,)
      - items: (batch_size,)
      - labels: (batch_size,)
      - memories_h: list of n_hop arrays, each (batch_size, n_memory)
      - memories_r: list of n_hop arrays, each (batch_size, n_memory)
      - memories_t: list of n_hop arrays, each (batch_size, n_memory)
    """
    batch = data[batch_indices]
    users = batch[:, 0]
    items = batch[:, 1]
    labels = batch[:, 2].astype(np.float32)

    memories_h = []
    memories_r = []
    memories_t = []

    for hop in range(n_hop):
        h_list = []
        r_list = []
        t_list = []
        for user_id in users:
            h_list.append(ripple_sets[user_id][hop][0])
            r_list.append(ripple_sets[user_id][hop][1])
            t_list.append(ripple_sets[user_id][hop][2])
        memories_h.append(np.array(h_list, dtype=np.int32))
        memories_r.append(np.array(r_list, dtype=np.int32))
        memories_t.append(np.array(t_list, dtype=np.int32))

    return {
        "users": users,
        "items": items,
        "labels": labels,
        "memories_h": memories_h,
        "memories_r": memories_r,
        "memories_t": memories_t,
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
    ripple_sets = construct_preference_sets(train_data, kg_dict, n_hop, n_memory)

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
        "ripple_sets": ripple_sets,
        "neighbor_entities": neighbor_entities,
        "neighbor_relations": neighbor_relations,
        "kg_dict": kg_dict,
    }
