#!/usr/bin/env python3
"""
Self-Supervised Learning Package - Main Startup Script

This script provides a unified interface to run self-supervised learning tasks
including training, fine-tuning, and evaluation of MAE, DINO v2, and other models.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the current directory to Python path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Self-Supervised Learning Package")

    # Main mode selection
    parser.add_argument(
        "mode",
        choices=["train", "finetune", "eval", "cluster"],
        help="Operation mode: train (self-supervised pre-training), "
        "finetune (fine-tune on downstream tasks), "
        "eval (evaluate models), "
        "or cluster (run clustering analysis)",
    )

    # Model type
    parser.add_argument(
        "--model",
        choices=["mae", "dino", "vae"],
        required=True,
        help="Model type to use",
    )

    # Operation-specific arguments
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--checkpoint", type=str, help="Pretrained checkpoint path")
    parser.add_argument("--dataset-path", type=str, help="Path to dataset")
    parser.add_argument(
        "--output-dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--task", type=str, help="Specific task (for clustering mode)")

    args = parser.parse_args()

    print(
        f"Running self-supervised learning in {args.mode} mode with {args.model} model"
    )

    if args.mode == "train":
        if args.model == "mae":
            print("Starting MAE self-supervised pre-training...")
            # We need to simulate command line args for the training function
            import sys

            from self_sup_learning.mae_finetune.mae_pretrain import (
                main as mae_train_main,
            )

            original_argv = sys.argv
            sys.argv = [
                "mae_pretrain.py",
                "--config",
                args.config,
                "--dataset-path",
                args.dataset_path,
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--epochs",
                str(args.epochs),
                "--work-dir",
                args.output_dir,
            ]
            mae_train_main()
            sys.argv = original_argv

        elif args.model == "dino":
            print("Starting DINO v2 self-supervised pre-training...")
            import sys

            from self_sup_learning.dino_finetune.dino_pretrain import (
                main as dino_train_main,
            )

            original_argv = sys.argv
            sys.argv = [
                "dino_pretrain.py",
                "--config",
                args.config,
                "--dataset-path",
                args.dataset_path,
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--epochs",
                str(args.epochs),
                "--work-dir",
                args.output_dir,
            ]
            dino_train_main()
            sys.argv = original_argv

        elif args.model == "vae":
            print("VAE training not implemented in this script")

    elif args.mode == "finetune":
        if args.model == "mae":
            print("Starting MAE fine-tuning...")
            import sys

            from self_sup_learning.mae_finetune.mae_finetune import (
                main as mae_finetune_main,
            )

            original_argv = sys.argv
            sys.argv = [
                "mae_finetune.py",
                "--config",
                args.config,
                "--checkpoint",
                args.checkpoint,
                "--dataset-path",
                args.dataset_path,
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--epochs",
                str(args.epochs),
                "--work-dir",
                args.output_dir,
            ]
            mae_finetune_main()
            sys.argv = original_argv

        elif args.model == "dino":
            print("Starting DINO v2 fine-tuning...")
            import sys

            from self_sup_learning.dino_finetune.dino_finetune import (
                main as dino_finetune_main,
            )

            original_argv = sys.argv
            sys.argv = [
                "dino_finetune.py",
                "--config",
                args.config,
                "--checkpoint",
                args.checkpoint,
                "--dataset-path",
                args.dataset_path,
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--epochs",
                str(args.epochs),
                "--work-dir",
                args.output_dir,
            ]
            dino_finetune_main()
            sys.argv = original_argv

    elif args.mode == "cluster":
        print(f"Running clustering analysis with {args.model} embeddings...")
        if args.task:
            if args.model == "mae" and args.task == "mae_clustering":
                print("Running MAE-based clustering analysis...")
                # Execute MAE clustering script
                import subprocess

                script_path = os.path.join(
                    os.path.dirname(__file__),
                    "mae_clustering",
                    "clustering_analysis.py",
                )
                if os.path.exists(script_path):
                    subprocess.run([sys.executable, script_path])
                else:
                    print(f"Clustering script not found at: {script_path}")

            elif args.model == "vae" and args.task == "vae_clustering":
                print("Running VAE-based clustering analysis...")
                # Execute VAE clustering script
                import subprocess

                script_path = os.path.join(
                    os.path.dirname(__file__),
                    "vae_clustering",
                    "imagenet_clustering.py",
                )
                if os.path.exists(script_path):
                    subprocess.run([sys.executable, script_path])
                else:
                    print(f"Clustering script not found at: {script_path}")

            else:
                print(
                    f"No specific clustering task implemented for {args.model} with task {args.task}"
                )
        else:
            print("Please specify a clustering task using --task parameter")

    elif args.mode == "eval":
        print(f"Evaluating {args.model} model...")
        # Evaluation implementation would go here
        print("Model evaluation functionality to be implemented")


if __name__ == "__main__":
    main()
