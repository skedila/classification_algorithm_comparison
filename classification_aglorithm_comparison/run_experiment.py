import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Load datasets (modify paths as needed)
def load_data():
    haberman = pd.read_csv("data/haberman.csv")
    wdbc = pd.read_csv("data/wdbc.csv")
    diabetes = pd.read_csv("data/diabetes.csv")
    return {"Haberman": haberman, "WDBC": wdbc, "Diabetes": diabetes}

# Preprocess datasets
def preprocess_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Define model hyperparameter grids
param_grids = {
    "SVM": {"C": np.logspace(-7, 3, 10), "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
    "LogisticRegression": {"C": np.logspace(-8, 4, 10)},
    "RandomForest": {"n_estimators": [100, 200, 300], "max_depth": [5, 10, 20], "min_samples_split": [2, 5]},
    "ANN": {"hidden_layer_sizes": [(50,), (100,)], "activation": ["relu", "tanh"], "solver": ["sgd", "adam"]}
}

models = {
    "SVM": SVC(probability=True),
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "ANN": MLPClassifier(max_iter=1000)
}

# Train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        grid_search = RandomizedSearchCV(model, param_grids[name], n_iter=10, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_prob),
            "APR": average_precision_score(y_test, y_prob)
        }
    return results

if __name__ == "__main__":
    datasets = load_data()
    for name, df in datasets.items():
        print(f"\nProcessing {name} dataset...")
        X_train, X_test, y_train, y_test = preprocess_data(df, df.columns[-1])
        results = train_and_evaluate(X_train, X_test, y_train, y_test)
        print(f"Results for {name}: {results}")
