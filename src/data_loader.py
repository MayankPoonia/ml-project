from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_data():
    """
    Loads the California Housing dataset and returns features and target as DataFrames.
    """
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    return X, y