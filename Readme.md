# House Price Prediction

This project predicts house prices using machine learning algorithms:

- Linear Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Regressor (CART)

We use the California Housing dataset from scikit-learn and compare the performance of the models.

---

## Project Structure

House_Price_Prediction/
├── data/ # Dataset and related files  
├── models/ # Trained models  
├── outputs/ # Evaluation results and plots  
├── src/ # Source code  
│ ├── data_loader.py # Loads and prepares the dataset  
│ ├── train.py # Trains the ML models  
│ ├── evaluate.py # Evaluates the models  
├── main.py # Entry point for running everything  
├── requirements.txt # Project dependencies  
└── README.md # Project info

---

## Setup Instructions

1. Clone the repository
2. Create a virtual environment
3. Install dependencies using pip-tools

```bash
git clone https://github.com/your-username/House_Price_Prediction.git
cd House_Price_Prediction
python3 -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
pip install pip-tools
pip-sync requirements.txt
```
