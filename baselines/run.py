import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .data_loader import load_baseline_data, precompute_history_sequences
from .models import LightGCN, NGCF, SASRec, BERT4Rec

def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline recommender models on MovieLens/Book-Crossing")
    parser.add_argument("--dataset", type=str, required=True, choices=["movie", "book"],
                        help="Dataset: 'movie' (MovieLens-1M) or 'book' (Book-Crossing)")
    parser.add_argument("--model", type=str, required=True, choices=["lightgcn", "ngcf", "sasrec", "bert4rec"],
                        help="Model to run")
    parser.add_argument("--dim", type=int, default=8, help="Embedding dimension d")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of layers for GCN models")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads for sequential models")
    parser.add_argument("--max_len", type=int, default=50, help="Max sequence length for sequential models")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: dataset specific)")
    parser.add_argument("--l2", type=float, default=1e-7, help="L2 regularization weight")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Max training epochs (default: dataset specific)")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    
    args = parser.parse_args()
    
    # Dataset specific defaults matching RKGCN/TS-RKGCN
    if args.epochs is None:
        args.epochs = 50 if args.dataset == "movie" else 20
        
    if args.lr is None:
        args.lr = 0.02 if args.dataset == "movie" else 0.01
        
    return args


def evaluate_baseline(model, data_split, model_type, adj_matrix_tf, history_seqs, batch_size=1024):
    from sklearn.metrics import roc_auc_score, accuracy_score
    all_predictions = []
    all_labels = []
    
    n_samples = len(data_split)
    indices = np.arange(n_samples)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        batch = data_split[batch_idx]
        
        batch_users = batch[:, 0].astype(np.int32)
        batch_items = batch[:, 1].astype(np.int32)
        batch_labels = batch[:, 2].astype(np.float32)
        
        if model_type in ["lightgcn", "ngcf"]:
            inputs = {
                "users": tf.constant(batch_users, dtype=tf.int32),
                "items": tf.constant(batch_items, dtype=tf.int32),
                "adj_matrix": adj_matrix_tf
            }
        else:
            batch_seqs = history_seqs[batch_idx]
            inputs = {
                "sequences": tf.constant(batch_seqs, dtype=tf.int32),
                "items": tf.constant(batch_items, dtype=tf.int32)
            }
            
        predictions = model(inputs, training=False)
        all_predictions.extend(predictions.numpy().tolist())
        all_labels.extend(batch_labels.tolist())
        
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    try:
        auc = roc_auc_score(all_labels, all_predictions)
    except ValueError:
        auc = 0.0
        
    binary_preds = (all_predictions >= 0.5).astype(int)
    acc = accuracy_score(all_labels, binary_preds)
    
    return auc, acc


def main():
    args = parse_args()
    
    # Setup data directory
    data_dir = os.path.join("datasets", "MovieLens-1M" if args.dataset == "movie" else "Book-Crossing")
    
    # Setup output directory
    dataset_out = "movielens-1m" if args.dataset == "movie" else "book-crossing"
    output_dir = os.path.join("outputs", "baselines", dataset_out, args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running baseline: {args.model.upper()} on {args.dataset.upper()} dataset")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load data
    data = load_baseline_data(data_dir)
    
    train_data = data["train_data"]
    val_data = data["val_data"]
    test_data = data["test_data"]
    n_users = data["n_users"]
    n_items = data["n_items"]
    n_entities = data["n_entities"]
    
    # Precompute historical sequences for sequential models
    if args.model in ["sasrec", "bert4rec"]:
        train_seqs = precompute_history_sequences(
            train_data, data["user_pos_items"], data["user_pos_times"], args.max_len, n_entities
        )
        val_seqs = precompute_history_sequences(
            val_data, data["user_pos_items"], data["user_pos_times"], args.max_len, n_entities
        )
        test_seqs = precompute_history_sequences(
            test_data, data["user_pos_items"], data["user_pos_times"], args.max_len, n_entities
        )
    else:
        train_seqs, val_seqs, test_seqs = None, None, None
        
    # Setup model
    if args.model == "lightgcn":
        model = LightGCN(n_users, n_entities, args.dim, args.n_layers, args.l2)
    elif args.model == "ngcf":
        model = NGCF(n_users, n_entities, args.dim, args.n_layers, args.l2)
    elif args.model == "sasrec":
        model = SASRec(n_entities, args.dim, args.max_len, args.num_heads, args.l2)
    elif args.model == "bert4rec":
        model = BERT4Rec(n_entities, args.dim, args.max_len, args.num_heads, args.l2)
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    # Optimizer
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
    patience_counter = 0
    n_train = len(train_data)
    
    # Model weight path
    weight_path = os.path.join(output_dir, "best_model.weights.h5")
    
    # Training Loop
    for epoch in range(1, args.epochs + 1):
        # Shuffle training data
        # To keep train_data aligned with train_seqs, we shuffle their indices instead!
        indices = np.arange(n_train)
        np.random.shuffle(indices)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            batch_idx = indices[start:end]
            batch = train_data[batch_idx]
            
            batch_users = batch[:, 0].astype(np.int32)
            batch_items = batch[:, 1].astype(np.int32)
            batch_labels = batch[:, 2].astype(np.float32)
            
            # Prepare inputs
            if args.model in ["lightgcn", "ngcf"]:
                inputs = {
                    "users": tf.constant(batch_users, dtype=tf.int32),
                    "items": tf.constant(batch_items, dtype=tf.int32),
                    "adj_matrix": data["adj_matrix"]
                }
            else:
                batch_seqs = train_seqs[batch_idx]
                inputs = {
                    "sequences": tf.constant(batch_seqs, dtype=tf.int32),
                    "items": tf.constant(batch_items, dtype=tf.int32)
                }
                
            labels = tf.constant(batch_labels, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                total_loss, _, _ = model.compute_loss(labels, predictions)
                
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += total_loss.numpy()
            n_batches += 1
            
        avg_loss = epoch_loss / n_batches
        
        # Evaluate on subset of train set for logging (max 10000 samples)
        if len(train_data) > 10000:
            eval_train_idx = np.random.choice(len(train_data), 10000, replace=False)
            eval_train = train_data[eval_train_idx]
            eval_train_seqs = train_seqs[eval_train_idx] if train_seqs is not None else None
        else:
            eval_train = train_data
            eval_train_seqs = train_seqs
            
        train_auc, train_acc = evaluate_baseline(
            model, eval_train, args.model, data["adj_matrix"],
            eval_train_seqs, args.batch_size
        )
        
        # Evaluate on validation set
        val_auc, val_acc = evaluate_baseline(
            model, val_data, args.model, data["adj_matrix"],
            val_seqs, args.batch_size
        )
        
        history["train_loss"].append(avg_loss)
        history["train_auc"].append(train_auc)
        history["train_acc"].append(train_acc)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"Train AUC: {train_auc:.4f}, ACC: {train_acc:.4f} | "
              f"Val AUC: {val_auc:.4f}, ACC: {val_acc:.4f}", end="")
              
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            model.save_weights(weight_path)
            print(" *best*")
        else:
            patience_counter += 1
            print()
            
        # Early stopping check
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered. No improvement for {args.patience} epochs.")
            break
            
    print(f"\nTraining complete. Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    
    # Restore best weights and evaluate on test set
    try:
        model.load_weights(weight_path)
        print("Restored best model weights.")
    except Exception:
        print("Warning: Could not restore best weights.")
        
    test_auc, test_acc = evaluate_baseline(
        model, test_data, args.model, data["adj_matrix"],
        test_seqs, args.batch_size
    )
    
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION ON TEST SET ({args.model.upper()})")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Test ACC: {test_acc:.4f}")
    print(f"{'='*60}\n")
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"Test ACC: {test_acc:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Val AUC: {best_val_auc:.4f}\n")
        
    # Plot curves
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    # Loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.title(f"Training Loss - {args.model.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()
    
    # AUC plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history["train_auc"], label="Train AUC")
    plt.plot(epochs_range, history["val_auc"], label="Val AUC")
    plt.title(f"AUC over Epochs - {args.model.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "auc_curve.png"))
    plt.close()
    
    # ACC plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history["train_acc"], label="Train ACC")
    plt.plot(epochs_range, history["val_acc"], label="Val ACC")
    plt.title(f"Accuracy over Epochs - {args.model.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()
    
    print("Baseline execution complete. Curves and metrics saved.")

if __name__ == "__main__":
    main()
