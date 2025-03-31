import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Task 1: Experiment with Different Decision Tree Depths
def experiment_tree_depths():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    depths = [2, 5, 10, None]  # None means no depth limit
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f'Depth {depth if depth is not None else "Unlimited"}: Accuracy = {acc:.2f}')

# Task 2: Use a Different Dataset
def train_on_wine():
    data = load_wine()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Wine Dataset - Decision Tree Accuracy: {acc:.2f}')
    print('Classification Report:\n', classification_report(y_test, y_pred))

    plt.figure(figsize=(15, 10))
    plot_tree(clf, feature_names=data.feature_names, class_names=list(data.target_names), filled=True)
    plt.show()

# Task 3: Compare Decision Tree with Random Forest
def compare_decision_tree_and_random_forest():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Decision Tree
    clf_tree = DecisionTreeClassifier(random_state=42)
    clf_tree.fit(X_train, y_train)
    y_pred_tree = clf_tree.predict(X_test)
    acc_tree = accuracy_score(y_test, y_pred_tree)
    
    # Random Forest
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    print(f'Decision Tree Accuracy: {acc_tree:.2f}')
    print(f'Random Forest Accuracy: {acc_rf:.2f}')
    print("\nRandom Forest performs better because it reduces overfitting by averaging multiple decision trees.")
    print('Classification Report - Decision Tree:\n', classification_report(y_test, y_pred_tree))
    print('Classification Report - Random Forest:\n', classification_report(y_test, y_pred_rf))

# Run all tasks
if __name__ == "__main__":
    experiment_tree_depths()
    train_on_wine()
    compare_decision_tree_and_random_forest()
