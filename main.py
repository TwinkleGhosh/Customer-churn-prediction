import pandas as pd

# Load Data
df = pd.read_csv("data/churn.csv")

# Basic checks
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())  # Check Missing Values

# Data Preprocessing

# 1. Convert totalcharges to Numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 2. Drop Missing Rows
df.dropna(inplace=True)

# 3. Convert target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 4. Drop customerID
df.drop("customerID", axis=1, inplace=True)

# 5. Encoding
df = pd.get_dummies(df, drop_first=True)

# Split Features & Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle Class Imbalance
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)


# Train Model

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Probability (Prediction + Threshold)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.3).astype(int)

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# FEATURE IMPORTANCE
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:")
print(feature_importance.sort_values(ascending=False))
# Save Model + Scaler
import pickle

churn_sample = X[y == 1].iloc[[0]]
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))
pickle.dump(churn_sample, open("sample_input.pkl", "wb"))
# Print sample probalilities
print("\nSample Probabilities:")
print(y_prob[:10])
