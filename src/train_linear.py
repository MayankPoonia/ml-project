from sklearn.linear_model import LinearRegression
import joblib
import os

def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/linear_model.joblib")
    print("[✔] Linear Regression model trained and saved.")
    return model