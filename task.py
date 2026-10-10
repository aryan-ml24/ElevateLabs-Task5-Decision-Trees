import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("\nDecision Tree Report:")
print(classification_report(y_test, dt_pred, target_names=data.target_names))

# Visualize Decision Tree
plt.figure(figsize=(18, 9))
plot_tree(dt, feature_names=data.feature_names,
          class_names=data.target_names, filled=True, fontsize=6)
plt.savefig("decision_tree.png", bbox_inches="tight")
plt.close()

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\nRandom Forest Report:")
print(classification_report(y_test, rf_pred, target_names=data.target_names))

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, rf_pred, display_labels=data.target_names
)
plt.title("Random Forest Confusion Matrix")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.close()

# Feature Importance
importance = pd.Series(
    rf.feature_importances_, index=data.feature_names
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

importance.head(10).sort_values().plot(kind="barh", figsize=(10, 6))
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", bbox_inches="tight")
plt.close()

# Cross Validation
scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
print("\nCross-validation scores:", scores)
print("Mean CV Accuracy:", scores.mean())

print("\nTask 5 Completed Successfully!")