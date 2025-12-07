# ml-playground

ML playground with monorepo setup using uv for dependency management.

## ⚠️ Security Notice

**This project uses PyTorch 1.13.1 which has known security vulnerabilities.**

Please read [SECURITY.md](SECURITY.md) for details about:
- Known vulnerabilities (heap overflow, use-after-free, RCE via torch.load)
- Security recommendations
- Migration path to secure versions

**DO NOT use this configuration in production environments without addressing the security vulnerabilities.**

### Tracking

- 📋 **Issue Template**: See `.github/ISSUE_TEMPLATE/security-pytorch-upgrade.md` to create a tracking issue
- 📝 **TODO**: See [TODO.md](TODO.md) for upgrade action items and blockers

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

## Submodules

- [mmpretrain](https://github.com/teepoy/mmpretrain)
- [mmdetection](https://github.com/teepoy/mmdetection)