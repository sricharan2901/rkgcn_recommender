"""
train.py — Training loop for RKGCN.

Handles:
  - Epoch-based training with mini-batch gradient descent
  - Validation evaluation after each epoch
  - Best model tracking by validation AUC
  - Logging of metrics per epoch
"""

import numpy as np
import tensorflow as tf
from .data_loader import get_batch_data
from .evaluate import evaluate


def train(model, data, args):
    """
    Train the RKGCN model.

    Args:
      model: RKGCN model instance
      data: dict from data_loader.load_data() containing all data structures
      args: argparse namespace with hyperparameters

    Returns:
      model: trained model
      history: dict with training metrics per epoch
    """
    train_data = data["train_data"]
    val_data = data["val_data"]
    ripple_sets = data["ripple_sets"]
    neighbor_entities = data["neighbor_entities"]
    neighbor_relations = data["neighbor_relations"]

    # Convert neighbor arrays to tensors (constant throughout training)
    neighbor_entities_tf = tf.constant(neighbor_entities, dtype=tf.int32)
    neighbor_relations_tf = tf.constant(neighbor_relations, dtype=tf.int32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    history = {
        "train_loss": [],
        "train_auc": [],
        "train_acc": [],
        "val_auc": [],
        "val_acc": [],
    }

    best_val_auc = 0.0
    best_epoch = 0
    n_train = len(train_data)

    print(f"\n{'='*60}")
    print(f"Starting training: {args.n_epoch} epochs, batch_size={args.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.n_epoch + 1):
        # Shuffle training data
        np.random.shuffle(train_data)
        indices = np.arange(n_train)

        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_kge_loss = 0.0
        epoch_l2_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            batch_idx = indices[start:end]

            batch = get_batch_data(batch_idx, train_data, ripple_sets, args.n_hop)

            # Prepare inputs
            inputs = {
                "items": tf.constant(batch["items"], dtype=tf.int32),
                "memories_h": [
                    tf.constant(h, dtype=tf.int32) for h in batch["memories_h"]
                ],
                "memories_r": [
                    tf.constant(r, dtype=tf.int32) for r in batch["memories_r"]
                ],
                "memories_t": [
                    tf.constant(t, dtype=tf.int32) for t in batch["memories_t"]
                ],
                "neighbor_entities": neighbor_entities_tf,
                "neighbor_relations": neighbor_relations_tf,
            }

            labels = tf.constant(batch["labels"], dtype=tf.float32)

            # Forward pass + compute gradients
            with tf.GradientTape() as tape:
                predictions, kge_loss = model(inputs, training=True)
                total_loss, rec_loss, kge_term, l2_term = model.compute_loss(
                    labels, predictions, kge_loss
                )

            # Backward pass
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_loss += total_loss.numpy()
            epoch_rec_loss += rec_loss.numpy()
            epoch_kge_loss += kge_term.numpy()
            epoch_l2_loss += l2_term.numpy()
            n_batches += 1

        # Average losses
        avg_loss = epoch_loss / n_batches
        avg_rec = epoch_rec_loss / n_batches
        avg_kge = epoch_kge_loss / n_batches
        avg_l2 = epoch_l2_loss / n_batches

        # Evaluate on training set (sample if too large)
        if len(train_data) > 10000:
            eval_train = train_data[np.random.choice(len(train_data), 10000, replace=False)]
        else:
            eval_train = train_data

        train_auc, train_acc = evaluate(
            model, eval_train, ripple_sets,
            neighbor_entities, neighbor_relations,
            args.n_hop, args.batch_size,
        )

        # Evaluate on validation set
        val_auc, val_acc = evaluate(
            model, val_data, ripple_sets,
            neighbor_entities, neighbor_relations,
            args.n_hop, args.batch_size,
        )

        # Record history
        history["train_loss"].append(avg_loss)
        history["train_auc"].append(train_auc)
        history["train_acc"].append(train_acc)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)

        # Track best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            # Save best weights
            model.save_weights("best_model_weights.h5")

        # Print epoch summary
        print(
            f"Epoch {epoch:3d}/{args.n_epoch} | "
            f"Loss: {avg_loss:.4f} (rec={avg_rec:.4f}, kge={avg_kge:.4f}, l2={avg_l2:.4f}) | "
            f"Train AUC: {train_auc:.4f}, ACC: {train_acc:.4f} | "
            f"Val AUC: {val_auc:.4f}, ACC: {val_acc:.4f}"
            + (" *best*" if val_auc == best_val_auc else "")
        )

    print(f"\nTraining complete. Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    # Restore best weights
    try:
        model.load_weights("best_model_weights.h5")
        print("Best model weights restored.")
    except Exception:
        print("Warning: Could not restore best weights.")

    return model, history
