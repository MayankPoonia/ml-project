from sklearn.preprocessing import StandardScaler

def preprocess_data(X):
    """
    Scales features using StandardScaler and returns the scaled data.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler