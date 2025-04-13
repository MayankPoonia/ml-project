from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"[📊] {model_name} - RMSE: {rmse:.2f} - R²: {r2:.2f}")

    # Create folder if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Save evaluation scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue', edgecolors='k')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/{model_name}_evaluation.png")
    plt.close()