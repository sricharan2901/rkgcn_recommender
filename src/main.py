"""
main.py — Main runner for RKGCN (Ripple Knowledge Graph Convolutional Networks).

Usage:
  # Step 1: Preprocess data (run once)
  python main.py --dataset movie --preprocess

  # Step 2: Train and evaluate
  python main.py --dataset movie

  # With custom hyperparameters
  python main.py --dataset book --dim 4 --n_hop 1 --n_memory 16 --lr 0.01 --l2 1e-5

Full list of arguments:
  --dataset:     'movie' (MovieLens-1M) or 'book' (Book-Crossing)
  --preprocess:  Run preprocessing only
  --dim:         Embedding dimension d (default: 8)
  --n_hop:       Number of preference propagation hops H (default: 2)
  --n_memory:    Preference set size per hop N_p (default: 32)
  --n_neighbor:  Neighbor set size for GCN N_e (default: 16)
  --kge_weight:  Weight for KGE loss (default: 0.01)
  --l2:          L2 regularization weight (default: 1e-7)
  --lr:          Learning rate (default: 0.02)
  --batch_size:  Training batch size (default: 1024)
  --n_epoch:     Number of training epochs (default: 10)
  --gcn_iter:    Number of GCN aggregation iterations (default: 1)
  --n_runs:      Number of runs to average (default: 1)
"""

import argparse
import os
import sys
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="RKGCN: Ripple Knowledge Graph Convolutional Networks for Recommendation"
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["movie", "book"],
        help="Dataset: 'movie' (MovieLens-1M) or 'book' (Book-Crossing)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data directory. Defaults to datasets/MovieLens-1M or datasets/Book-Crossing",
    )

    # Mode
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run data preprocessing only (no training)",
    )

    # Model hyperparameters
    parser.add_argument("--dim", type=int, default=8, help="Embedding dimension d")
    parser.add_argument(
        "--n_hop", type=int, default=2, help="Number of preference propagation hops H"
    )
    parser.add_argument(
        "--n_memory", type=int, default=32, help="Preference set size per hop N_p"
    )
    parser.add_argument(
        "--n_neighbor", type=int, default=16, help="Neighbor set size for GCN N_e"
    )
    parser.add_argument(
        "--kge_weight", type=float, default=0.01, help="Weight for KGE loss"
    )
    parser.add_argument(
        "--l2", type=float, default=1e-7, help="L2 regularization weight"
    )
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Training batch size"
    )
    parser.add_argument(
        "--n_epoch", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--gcn_iter", type=int, default=1, help="Number of GCN aggregation iterations"
    )
    parser.add_argument(
        "--n_runs", type=int, default=1, help="Number of runs to average results"
    )

    args = parser.parse_args()

    if args.data_dir is None:
        if args.dataset == "movie":
            args.data_dir = os.path.join("datasets", "MovieLens-1M")
        else:
            args.data_dir = os.path.join("datasets", "Book-Crossing")

    return args


def run_single(args, run_id=1):
    """Run a single training + evaluation cycle."""
    import tensorflow as tf
    from .data_loader import load_data
    from .model import RKGCN
    from .train import train
    from .evaluate import evaluate

    print(f"\n{'#'*60}")
    print(f"# Run {run_id}")
    print(f"{'#'*60}")

    # Set random seeds
    seed = 42 + run_id
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load data
    data = load_data(
        data_dir=args.data_dir,
        n_hop=args.n_hop,
        n_memory=args.n_memory,
        n_neighbor=args.n_neighbor,
    )

    # Build model
    model = RKGCN(
        n_entities=data["n_entities"],
        n_relations=data["n_relations"],
        dim=args.dim,
        n_hop=args.n_hop,
        n_memory=args.n_memory,
        n_neighbor=args.n_neighbor,
        kge_weight=args.kge_weight,
        l2_weight=args.l2,
        gcn_iter=args.gcn_iter,
    )

    print(f"\nModel: RKGCN")
    print(f"  Entities: {data['n_entities']}, Relations: {data['n_relations']}")
    print(f"  Embedding dim: {args.dim}, Hops: {args.n_hop}")
    print(f"  Memory size: {args.n_memory}, Neighbor size: {args.n_neighbor}")
    print(f"  GCN iterations: {args.gcn_iter}")
    print(f"  LR: {args.lr}, L2: {args.l2}, KGE weight: {args.kge_weight}")
    print(f"  Batch size: {args.batch_size}, Epochs: {args.n_epoch}")

    # Train
    model, history = train(model, data, args)

    # Final evaluation on test set
    print(f"\n{'='*60}")
    print(f"Final Evaluation on Test Set")
    print(f"{'='*60}")

    test_auc, test_acc = evaluate(
        model,
        data["test_data"],
        data["ripple_sets"],
        data["neighbor_entities"],
        data["neighbor_relations"],
        args.n_hop,
        args.batch_size,
    )

    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Test ACC: {test_acc:.4f}")

    return test_auc, test_acc, history


def main():
    args = parse_args()

    # Print configuration
    print(f"\n{'='*60}")
    print(f"RKGCN — Ripple Knowledge Graph Convolutional Networks")
    print(f"{'='*60}")
    print(f"Dataset:    {args.dataset}")
    print(f"Data dir:   {args.data_dir}")
    print()

    # Preprocessing mode
    if args.preprocess:
        from .preprocess import preprocess

        preprocess(args.dataset, args.data_dir)
        print("Preprocessing complete. Run without --preprocess to train.")
        return

    # Check if preprocessed data exists
    if not os.path.exists(os.path.join(args.data_dir, "ratings_final.txt")):
        print(f"ERROR: Preprocessed data not found in {args.data_dir}/")
        print(f"Run preprocessing first:")
        print(f"  python main.py --dataset {args.dataset} --preprocess")
        sys.exit(1)

    if not os.path.exists(os.path.join(args.data_dir, "kg_final.txt")):
        print(f"ERROR: Knowledge graph file not found in {args.data_dir}/")
        print(f"Make sure kg_final.txt exists in {args.data_dir}/")
        sys.exit(1)

    # Run training and evaluation
    all_auc = []
    all_acc = []

    for run_id in range(1, args.n_runs + 1):
        test_auc, test_acc, history = run_single(args, run_id)
        all_auc.append(test_auc)
        all_acc.append(test_acc)

    # Report final results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({args.n_runs} run(s))")
    print(f"{'='*60}")
    print(f"  Test AUC: {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
    print(f"  Test ACC: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
