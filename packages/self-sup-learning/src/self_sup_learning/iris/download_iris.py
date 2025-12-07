#!/usr/bin/env python3
"""
Script to download and explore the Iris dataset
"""

import os

import pandas as pd
from sklearn.datasets import load_iris


def download_iris_dataset():
    """Download the iris dataset and save it in different formats"""

    # Load the iris dataset
    iris = load_iris()

    # Create a pandas DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = df["target"].map(
        {i: name for i, name in enumerate(iris.target_names)}
    )

    # Create directory to store the dataset
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save as CSV
    csv_path = os.path.join(data_dir, "iris.csv")
    df.to_csv(csv_path, index=False)
    print(f"Iris dataset saved as CSV: {csv_path}")

    # Save as JSON
    json_path = os.path.join(data_dir, "iris.json")
    df.to_json(json_path, orient="records", indent=2)
    print(f"Iris dataset saved as JSON: {json_path}")

    # Display basic information about the dataset
    print("\nDataset Information:")
    print(f"Shape: {df.shape}")
    print(f"Features: {list(iris.feature_names)}")
    print(f"Target names: {list(iris.target_names)}")
    print("First 5 rows:")
    print(df.head())

    # Show class distribution
    print("\nClass distribution:")
    print(df["species"].value_counts())

    return df


if __name__ == "__main__":
    print("Downloading Iris dataset...")
    df = download_iris_dataset()
    print("\nDownload completed successfully!")
