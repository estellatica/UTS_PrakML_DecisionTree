import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

def load_data(path):
    df = pd.read_csv(path)
    # Encode label: orange->0, grapefruit->1
    df['label'] = df['name'].map({'orange': 0, 'grapefruit': 1})
    return df

def preprocess(df):
    # Tangani missing (jika ada)
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.mean())
    # Fitur: semua kecuali 'name' dan 'label'
    X = df.drop(['name', 'label'], axis=1)
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def tune_and_train(X_train, y_train):
    dt = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [None, 3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    best_dt = grid.best_estimator_
    best_dt.fit(X_train, y_train)
    return best_dt

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['orange','grapefruit']))
    return confusion_matrix(y_test, y_pred)

def plot_results(model, cm, feature_names):
    plt.figure(figsize=(12,8))
    plot_tree(model, feature_names=feature_names, class_names=['orange','grapefruit'], filled=True, rounded=True)
    plt.title("Decision Tree")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ['orange','grapefruit'])
    plt.yticks(ticks, ['orange','grapefruit'])
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def main():
    df = load_data("citrus.csv")
    X_train, X_test, y_train, y_test = preprocess(df)
    model = tune_and_train(X_train, y_train)
    cm = evaluate(model, X_test, y_test)
    plot_results(model, cm, X_train.columns)
    print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")

if __name__ == "__main__":
    main()
