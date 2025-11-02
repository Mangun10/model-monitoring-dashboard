"""
Generate sample datasets for testing the ML model monitoring dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import os

def create_classification_dataset(n_samples=1000, n_features=10, n_classes=2, random_state=42):
    """
    Create a sample classification dataset.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features//2,
        n_redundant=n_features//4,
        n_classes=n_classes,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_names

def create_regression_dataset(n_samples=1000, n_features=10, random_state=42):
    """
    Create a sample regression dataset.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features//2,
        noise=0.1,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, feature_names

def save_dataset_to_csv(X, y, feature_names, filename, target_name='target'):
    """
    Save dataset to CSV file.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        filename: Output filename
        target_name: Name for target column
    """
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df[target_name] = y
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"✅ Dataset saved to: {filename}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {feature_names}")
    print(f"   Target: {target_name}")

def create_sample_datasets():
    """Create and save sample datasets for testing."""
    
    print("🔄 Creating sample datasets...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Create classification dataset
    print("\n📊 Creating classification dataset...")
    X_train_cls, X_test_cls, y_train_cls, y_test_cls, features_cls = create_classification_dataset()
    
    # Save classification datasets
    save_dataset_to_csv(X_train_cls, y_train_cls, features_cls, 'data/classification_train.csv', 'is_positive')
    save_dataset_to_csv(X_test_cls, y_test_cls, features_cls, 'data/classification_test.csv', 'is_positive')
    
    # Create regression dataset
    print("\n📈 Creating regression dataset...")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, features_reg = create_regression_dataset()
    
    # Save regression datasets
    save_dataset_to_csv(X_train_reg, y_train_reg, features_reg, 'data/regression_train.csv', 'target_value')
    save_dataset_to_csv(X_test_reg, y_test_reg, features_reg, 'data/regression_test.csv', 'target_value')
    
    print("\n✅ All sample datasets created successfully!")
    print("\nDatasets created:")
    print("   - data/classification_train.csv")
    print("   - data/classification_test.csv")
    print("   - data/regression_train.csv")
    print("   - data/regression_test.csv")

if __name__ == "__main__":
    create_sample_datasets()