from sklearn.tree import DecisionTreeRegressor
import joblib
import os

def train_cart_model(X_train, y_train, max_depth=None):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/cart_model.joblib")
    print("[✔] CART (Decision Tree) model trained and saved.")
    return model