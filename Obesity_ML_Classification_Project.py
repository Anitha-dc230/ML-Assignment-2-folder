# Obesity Dataset - End-to-End ML Classification Project
This notebook includes:
- EDA
- Preprocessing
- Train-Test Split
- Implementation of 6 ML Models
- Evaluation Metrics Comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load Dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
df.head()

# Basic EDA
print("Training Dataset size:", df.shape)
print("\nTraining Dataset info:\n", df.info)

print("\nData Types:\n",df.dtypes)

print("\nStatistics\n",df.describe())

print("\nMissing values in dataset:\n",df.isnull().sum())

# Boxplot: Age vs NObeyesdad
plt.figure()
df.boxplot(column='Age', by='NObeyesdad')
plt.title('Age vs NObeyesdad')
plt.suptitle('')
plt.xlabel('NObeyesdad')
plt.ylabel('Age')
plt.xticks(rotation=45)
plt.show()

# Boxplot: Height vs NObeyesdad
plt.figure()
df.boxplot(column='Height', by='NObeyesdad')
plt.title('Height vs NObeyesdad')
plt.suptitle('')
plt.xlabel('NObeyesdad')
plt.ylabel('Height')
plt.xticks(rotation=45)
plt.show()

# Boxplot: Weight vs NObeyesdad
plt.figure()
df.boxplot(column='Weight', by='NObeyesdad')
plt.title('Weight vs NObeyesdad')
plt.suptitle('')
plt.xlabel('NObeyesdad')
plt.ylabel('Weight')
plt.xticks(rotation=45)
plt.show()

# Scatter: Height vs Weight
plt.figure()
plt.scatter(df['Height'], df['Weight'])
plt.title('Height vs Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

# Encode categorical variables
label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

#df['NObeyesdad_encoded'] = df['NObeyesdad']

# Compute correlation with target
corr_matrix = df.corr(numeric_only=True)
target_corr = corr_matrix['NObeyesdad'].drop('NObeyesdad')

# Sort by absolute correlation
target_corr_sorted = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)

print("Feature Correlation with Target (sorted by importance):")
print(target_corr_sorted)

# Plot correlation values
plt.figure()
plt.bar(target_corr_sorted.index, target_corr_sorted.values)
plt.xticks(rotation=90)
plt.title("Feature Correlation with Encoded Target")
plt.xlabel("Features")
plt.ylabel("Correlation")
plt.show()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=0.3)
plt.title("Correlation Heatmap of Obesity dataset Features")
plt.show()

# Define Features & Target
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Dataset size:", X_train.shape)
print("Testing Dataset size:", X_test.shape)

print("Training Data columns:", X_train.columns)

results = {}

# Logistic Regression
lr = LogisticRegression(max_iter=500)
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)
y_prob = lr.predict_proba(X_test_scaled)

results['Logistic Regression'] = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(label_binarize(y_test, classes=np.unique(y)), y_prob, multi_class='ovr'),
    "Precision": precision_score(y_test, y_pred, average='weighted'),
    "Recall": recall_score(y_test, y_pred, average='weighted'),
    "F1 Score": f1_score(y_test, y_pred, average='weighted'),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)

results['Decision Tree'] = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(label_binarize(y_test, classes=np.unique(y)), y_prob, multi_class='ovr'),
    "Precision": precision_score(y_test, y_pred, average='weighted'),
    "Recall": recall_score(y_test, y_pred, average='weighted'),
    "F1 Score": f1_score(y_test, y_pred, average='weighted'),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
y_prob = knn.predict_proba(X_test_scaled)

results['KNN'] = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(label_binarize(y_test, classes=np.unique(y)), y_prob, multi_class='ovr'),
    "Precision": precision_score(y_test, y_pred, average='weighted'),
    "Recall": recall_score(y_test, y_pred, average='weighted'),
    "F1 Score": f1_score(y_test, y_pred, average='weighted'),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)

results['Naive Bayes'] = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(label_binarize(y_test, classes=np.unique(y)), y_prob, multi_class='ovr'),
    "Precision": precision_score(y_test, y_pred, average='weighted'),
    "Recall": recall_score(y_test, y_pred, average='weighted'),
    "F1 Score": f1_score(y_test, y_pred, average='weighted'),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_importance_df = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

print("\nRandom Forest Feature Importance (sorted):")
print(feature_importance_df)

# Plot Feature Importance
plt.figure()
plt.bar(feature_importance_df.index, feature_importance_df.values)
plt.xticks(rotation=90)
plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()

# Display top important features
top_features = feature_importance_df.head(5)
print("\nTop 5 Most Informative Features:")
print(top_features)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

results['Random Forest'] = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(label_binarize(y_test, classes=np.unique(y)), y_prob, multi_class='ovr'),
    "Precision": precision_score(y_test, y_pred, average='weighted'),
    "Recall": recall_score(y_test, y_pred, average='weighted'),
    "F1 Score": f1_score(y_test, y_pred, average='weighted'),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

# XGBoost
xgb = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)

results['XGBoost'] = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(label_binarize(y_test, classes=np.unique(y)), y_prob, multi_class='ovr'),
    "Precision": precision_score(y_test, y_pred, average='weighted'),
    "Recall": recall_score(y_test, y_pred, average='weighted'),
    "F1 Score": f1_score(y_test, y_pred, average='weighted'),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

# Compare Results
results_df = pd.DataFrame(results).T
results_df

import joblib
joblib.dump(lr, "model/logistic_regression_model.pkl")
joblib.dump(dt, "model/decision_tree_model.pkl")
joblib.dump(knn, "model/knn_model.pkl")
joblib.dump(nb, "model/naive_bayes_model.pkl")
joblib.dump(rf, "model/random_forest_model.pkl")
joblib.dump(xgb, "model/xgboost_model.pkl")
joblib.dump(scaler, "model/preprocessing_pipeline.pkl")

