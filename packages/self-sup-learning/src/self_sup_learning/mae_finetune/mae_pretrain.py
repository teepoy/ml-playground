"""
Self-supervised pre-training script for MAE (Masked Autoencoders) models
Based on mmpretrain MAE implementation
"""

import argparse

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="MAE Pre-training")
    parser.add_argument("--config", help="train config file path", required=True)
    parser.add_argument("--work-dir", help="the directory to save logs and models")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="batch size")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument(
        "--dataset-path", type=str, help="path to the dataset", required=True
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print(f"Starting MAE pre-training with config: {args.config}")
    print(f"Dataset path: {args.dataset_path}")

    # Load the model configuration
    cfg = Config.fromfile(args.config)

    # Update config with command line arguments
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.batch_size:
        cfg.train_dataloader.batch_size = args.batch_size
        cfg.val_dataloader.batch_size = args.batch_size
        cfg.test_dataloader.batch_size = args.batch_size

    # Update the learning rate based on linear scaling rule (batch_size / 256 * original_lr)
    if hasattr(cfg, "optim_wrapper") and hasattr(cfg.optim_wrapper, "optimizer"):
        scaled_lr = args.lr * args.batch_size / 256
        cfg.optim_wrapper.optimizer.lr = scaled_lr
        print(
            f"Original LR: {args.lr}, Scaled LR: {scaled_lr} (based on batch size {args.batch_size})"
        )

    # Update the number of epochs
    if hasattr(cfg, "train_cfg"):
        cfg.train_cfg.max_epochs = args.epochs

    # Set the dataset path
    if hasattr(cfg.train_dataloader.dataset, "data_root"):
        cfg.train_dataloader.dataset.data_root = args.dataset_path
    if hasattr(cfg.val_dataloader.dataset, "data_root"):
        cfg.val_dataloader.dataset.data_root = args.dataset_path
    if hasattr(cfg.test_dataloader.dataset, "data_root"):
        cfg.test_dataloader.dataset.data_root = args.dataset_path

    # Initialize and start the trainer
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
