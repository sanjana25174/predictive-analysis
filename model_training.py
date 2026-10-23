import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the cleaned dataset
df = pd.read_csv("ai4i2020_cleaned.csv")

# Display dataset information
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
# Separate input features (X) and target (y)
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

print("\nInput features (X):")
print(X.columns.tolist())

print("\nTarget (y):")
print("Machine failure")

print("\nX shape:", X.shape)
print("y shape:", y.shape)
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")
print("Scaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)
# Train Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_logistic = logistic_model.predict(X_test_scaled)

# Get failure probabilities
y_prob_logistic = logistic_model.predict_proba(X_test_scaled)[:, 1]

print("\nLogistic Regression training completed!")
print("Number of predictions:", len(y_pred_logistic))
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
logistic_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_logistic = logistic_model.predict(X_test_scaled)

# Get failure probabilities
y_prob_logistic = logistic_model.predict_proba(X_test_scaled)[:, 1]

print("\nLogistic Regression training completed!")
print("Number of predictions:", len(y_pred_logistic))
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
# Evaluate Logistic Regression
accuracy = accuracy_score(y_test, y_pred_logistic)
precision = precision_score(y_test, y_pred_logistic, zero_division=0)
recall = recall_score(y_test, y_pred_logistic, zero_division=0)
f1 = f1_score(y_test, y_pred_logistic, zero_division=0)
cm = confusion_matrix(y_test, y_pred_logistic)

print("\nLogistic Regression Evaluation:")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))

print("\nConfusion Matrix:")
print(cm)
# Train Decision Tree model
decision_tree_model = DecisionTreeClassifier(
    max_depth=5,
    class_weight="balanced",
    random_state=42
)

decision_tree_model.fit(X_train, y_train)

# Make predictions
y_pred_tree = decision_tree_model.predict(X_test)

# Get failure probabilities
y_prob_tree = decision_tree_model.predict_proba(X_test)[:, 1]

print("\nDecision Tree training completed!")
print("Number of predictions:", len(y_pred_tree))

# Evaluate Decision Tree
accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree, zero_division=0)
recall_tree = recall_score(y_test, y_pred_tree, zero_division=0)
f1_tree = f1_score(y_test, y_pred_tree, zero_division=0)
cm_tree = confusion_matrix(y_test, y_pred_tree)

print("\nDecision Tree Evaluation:")
print("Accuracy :", round(accuracy_tree, 4))
print("Precision:", round(precision_tree, 4))
print("Recall   :", round(recall_tree, 4))
print("F1-score :", round(f1_tree, 4))

print("\nDecision Tree Confusion Matrix:")
print(cm_tree)
# Train Random Forest model
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = random_forest_model.predict(X_test)

# Get failure probabilities
y_prob_rf = random_forest_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest training completed!")
print("Number of predictions:", len(y_pred_rf))

# Evaluate Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
cm_rf = confusion_matrix(y_test, y_pred_rf)

print("\nRandom Forest Evaluation:")
print("Accuracy :", round(accuracy_rf, 4))
print("Precision:", round(precision_rf, 4))
print("Recall   :", round(recall_rf, 4))
print("F1-score :", round(f1_rf, 4))

print("\nRandom Forest Confusion Matrix:")
print(cm_rf)
# Save the trained Random Forest model
joblib.dump(random_forest_model, "random_forest_model.pkl")

# Save the feature names
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("\nRandom Forest model saved successfully!")
print("Feature names saved successfully!")