from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train_linear import train_linear_model
from src.train_knn import train_knn_model
from src.train_cart import train_cart_model
from src.evaluate import evaluate_model

def main():
    # Load and split data
    X, y = load_data()
    X_scaled, scaler = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train models
    linear_model = train_linear_model(X_train, y_train)
    knn_model = train_knn_model(X_train, y_train)
    cart_model = train_cart_model(X_train, y_train)

    # Evaluate models
    evaluate_model(linear_model, X_test, y_test, "LinearRegression")
    evaluate_model(knn_model, X_test, y_test, "KNN")
    evaluate_model(cart_model, X_test, y_test, "CART")

if __name__ == "__main__":
    main()