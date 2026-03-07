"""
evaluate.py — Evaluation metrics for RKGCN.

Computes AUC and Accuracy on a given dataset split.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def evaluate(model, data, ripple_sets, neighbor_entities, neighbor_relations,
             n_hop, batch_size=1024):
    """
    Evaluate the RKGCN model on a dataset split.

    Args:
      model: trained RKGCN model
      data: numpy array (N, 3) with [user_id, item_id, label]
      ripple_sets: dict of user ripple sets
      neighbor_entities: (n_entities, n_neighbor) array
      neighbor_relations: (n_entities, n_neighbor) array
      n_hop: number of hops
      batch_size: evaluation batch size

    Returns:
      auc: Area Under ROC Curve
      acc: Classification accuracy (threshold=0.5)
    """
    import tensorflow as tf
    from .data_loader import get_batch_data

    all_predictions = []
    all_labels = []

    n_samples = len(data)
    indices = np.arange(n_samples)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]

        batch = get_batch_data(batch_idx, data, ripple_sets, n_hop)

        inputs = {
            "items": tf.constant(batch["items"], dtype=tf.int32),
            "memories_h": [tf.constant(h, dtype=tf.int32) for h in batch["memories_h"]],
            "memories_r": [tf.constant(r, dtype=tf.int32) for r in batch["memories_r"]],
            "memories_t": [tf.constant(t, dtype=tf.int32) for t in batch["memories_t"]],
            "neighbor_entities": tf.constant(neighbor_entities, dtype=tf.int32),
            "neighbor_relations": tf.constant(neighbor_relations, dtype=tf.int32),
        }

        predictions, _ = model(inputs, training=False)
        all_predictions.extend(predictions.numpy().tolist())
        all_labels.extend(batch["labels"].tolist())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # AUC
    try:
        auc = roc_auc_score(all_labels, all_predictions)
    except ValueError:
        auc = 0.0  # If only one class present

    # Accuracy
    binary_preds = (all_predictions >= 0.5).astype(int)
    acc = accuracy_score(all_labels, binary_preds)

    return auc, acc
