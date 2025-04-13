from sklearn.neighbors import KNeighborsRegressor
import joblib
import os

def train_knn_model(X_train, y_train, k=5):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/knn_model.joblib")
    print("[✔] KNN model trained and saved.")
    return model