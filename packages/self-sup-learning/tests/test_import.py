"""
Basic tests to ensure the package structure is correct.

These tests verify:
- Package can be imported
- All submodules are accessible
- Version information is available
"""

import pytest


def test_package_import():
    """Test that the main package can be imported."""
    import self_sup_learning

    assert self_sup_learning is not None


def test_package_version():
    """Test that package version is defined."""
    import self_sup_learning

    assert hasattr(self_sup_learning, "__version__")
    assert self_sup_learning.__version__ == "0.1.0"


def test_package_all():
    """Test that __all__ is defined correctly."""
    import self_sup_learning

    assert hasattr(self_sup_learning, "__all__")
    expected_modules = [
        "dino",
        "dino_finetune",
        "mae_finetune",
        "mae_clustering",
        "vae",
        "vae_clustering",
        "iris",
    ]
    assert set(self_sup_learning.__all__) == set(expected_modules)


def test_dino_import():
    """Test that dino module can be imported."""
    from self_sup_learning import dino

    assert dino is not None


def test_dino_finetune_import():
    """Test that dino_finetune module can be imported."""
    from self_sup_learning import dino_finetune

    assert dino_finetune is not None


def test_mae_finetune_import():
    """Test that mae_finetune module can be imported."""
    from self_sup_learning import mae_finetune

    assert mae_finetune is not None


def test_mae_clustering_import():
    """Test that mae_clustering module can be imported."""
    from self_sup_learning import mae_clustering

    assert mae_clustering is not None


def test_vae_import():
    """Test that vae module can be imported."""
    from self_sup_learning import vae

    assert vae is not None


def test_vae_clustering_import():
    """Test that vae_clustering module can be imported."""
    from self_sup_learning import vae_clustering

    assert vae_clustering is not None


def test_iris_import():
    """Test that iris module can be imported."""
    from self_sup_learning import iris

    assert iris is not None


def test_all_modules_have_init():
    """Test that all submodules have __init__.py files."""
    from self_sup_learning import (
        dino,
        dino_finetune,
        iris,
        mae_clustering,
        mae_finetune,
        vae,
        vae_clustering,
    )

    modules = [dino, dino_finetune, mae_finetune, mae_clustering, vae, vae_clustering, iris]

    for module in modules:
        # Each module should be importable and not None
        assert module is not None
