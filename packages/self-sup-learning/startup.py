#!/usr/bin/env python3
"""
Self-Supervised Learning Package - Main Entry Point

Usage:
  python startup.py train --model mae --config /path/to/config --dataset-path /path/to/dataset
  python startup.py finetune --model dino --config /path/to/config --checkpoint /path/to/checkpoint --dataset-path /path/to/dataset
  python startup.py cluster --model mae --task mae_clustering
"""

import os
import sys

# Add the self_sup_learning package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from startup_script import main

if __name__ == "__main__":
    main()
