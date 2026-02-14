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
