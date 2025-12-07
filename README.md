# ml-playground

ML playground with monorepo setup using uv for dependency management.

## ✅ Security Status

**This project now uses PyTorch 2.9.1** which addresses all previously identified security vulnerabilities.

### Vulnerabilities Fixed:
- ✅ Heap buffer overflow (was in < 2.2.0, patched in 2.2.0)
- ✅ Use-after-free (was in < 2.2.0, patched in 2.2.0)
- ✅ Remote code execution via torch.load (was in < 2.6.0, patched in 2.6.0)

**Note**: The project was upgraded from PyTorch 1.13.1 to 2.9.1 to address critical security vulnerabilities. The mmpretrain and mmdetection submodules may require updates to work with PyTorch 2.9+. Please verify compatibility before using the submodules.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management and is configured for Linux environments only.

### Requirements

- Python 3.8-3.10 (PyTorch 1.13.1 compatibility)
- Linux OS (configured in pyproject.toml)
- uv package manager

### Installation

1. Install uv:
```bash
pip install uv
```

2. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/teepoy/ml-playground.git
cd ml-playground
```

Or if already cloned, initialize submodules:
```bash
git submodule update --init --recursive
```

3. Install dependencies:
```bash
uv sync
```

4. Run the project:
```bash
uv run python main.py
```

## Project Structure

```
ml-playground/
├── packages/
│   ├── coco_s3_loader/  # PyTorch COCO dataloader with S3 support
│   ├── mmpretrain/      # MMPretrain submodule
│   └── mmdetection/     # MMDetection submodule
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock             # Dependency lock file
└── main.py             # Main entry point
```

## Dependencies

- PyTorch 1.13.1
- torchvision 0.14.1
- Additional packages: see pyproject.toml

## Packages

### COCO S3 Loader

A PyTorch-compatible dataloader for loading COCO datasets with S3 URLs as image paths.

**Features:**
- Load images directly from S3 URLs in COCO JSON annotations
- Error handling for inaccessible S3 URLs (skip, raise, or return None)
- Support for PyTorch transforms
- Full DataLoader batching support
- Comprehensive test coverage

**Quick Start:**
```bash
cd packages/coco_s3_loader
uv sync
uv run pytest  # Run tests
```

See [packages/coco_s3_loader/README.md](packages/coco_s3_loader/README.md) for detailed documentation.

## Submodules

- [mmpretrain](https://github.com/teepoy/mmpretrain)
- [mmdetection](https://github.com/teepoy/mmdetection)