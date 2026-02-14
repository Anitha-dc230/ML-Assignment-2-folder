# ML-Assignment-2-folder
ML assignment - 2

README.md
README.md_
a. Problem Statement

Obesity is a major public health concern worldwide. Early identification of obesity levels based on lifestyle habits, physical attributes, and eating behaviors can help in prevention and treatment planning.

The objective of this project is to build and compare multiple machine learning classification models to predict the obesity level (NObeyesdad) of individuals using demographic, physical, and lifestyle-related features.

The project implements six classification algorithms and evaluates their performance using multiple evaluation metrics.

b. Dataset Description The dataset used in this project is:

Obesity Dataset (ObesityDataSet_raw_and_data_sinthetic.csv) Target Variable: NObeyesdad — Obesity level category (Multiclass classification)

Feature Categories:

Demographic Features
Age

Gender

Height

Weight

Eating Habits
FAVC (Frequent high-calorie food consumption)

FCVC (Vegetable consumption frequency)

NCP (Number of main meals)

CAEC (Food between meals)

CH2O (Water consumption)

CALC (Alcohol consumption)

Lifestyle Habits
FAF (Physical activity frequency)

TUE (Time using technology devices)

SMOKE (Smoking habit)

SCC (Calories monitoring)

MTRANS (Transportation used)

Preprocessing Performed:

Label encoding and One-hot encoding

Feature scaling (StandardScaler)

Train-test split (80:20, stratified)

This is a multiclass classification problem.

c. Models Used All six models were implemented on the same dataset: Evaluation Metrics Used:

Accuracy

AUC Score (One-vs-Rest for multiclass)

Precision (Weighted)

Recall (Weighted)

F1 Score (Weighted)

Matthews Correlation Coefficient (MCC)

Logistic Regression
Logistic Regression is a linear classification model suitable for multiclass problems using the One-vs-Rest strategy.

Metrics Reported: Accuracy,AUC Score,Precision,Recall,F1 Score,MCC

Observation: Performs well for linearly separable patterns. Moderate performance compared to ensemble models.

Decision Tree Classifier
Decision Tree is a non-linear model that splits data based on feature importance.

Metrics Reported: Accuracy,AUC Score,Precision,Recall,F1 Score,MCC

Observation: Capable of capturing non-linear relationships but may overfit without pruning.

K-Nearest Neighbor (KNN)
KNN is a distance-based algorithm that classifies based on nearest data points.

Metrics Reported: Accuracy,AUC Score,Precision,Recall,F1 Score,MCC

Observation: Sensitive to feature scaling. Performance improves after normalization.

Naive Bayes (Gaussian)
Naive Bayes assumes conditional independence among features.

Metrics Reported: Accuracy,AUC Score,Precision,Recall,F1 Score,MCC

Observation: Fast and simple model. Performance may be lower due to independence assumption.

Random Forest (Ensemble Model)
Random Forest combines multiple decision trees using bagging to improve performance.

Metrics Reported: Accuracy,AUC Score,Precision,Recall,F1 Score,MCC

Observation: Strong performance across all metrics. Handles non-linearity well and reduces overfitting.

XGBoost (Ensemble Model)
XGBoost is a gradient boosting algorithm that builds trees sequentially to reduce errors.

Metrics Reported: Accuracy,AUC Score,Precision,Recall,F1 Score,MCC

Observation: Typically achieves the best performance. High AUC and F1 score due to boosting mechanism.

Conclusion:

Ensemble models (Random Forest and XGBoost) achieved the highest performance. Logistic Regression and KNN performed moderately well. Naive Bayes showed comparatively lower performance.

Model Comparison Table:

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.867	0.985	0.867	0.867	0.865	0.846
Decision Tree	0.917	0.950	0.920	0.917	0.918	0.903
kNN	0.806	0.946	0.813	0.806	0.798	0.777
Naive Bayes	0.602	0.907	0.645	0.602	0.579	0.547
Random Forest (Ensemble)	0.957	0.997	0.960	0.957	0.958	0.950
XGBoost (Ensemble)	0.957	0.996	0.959	0.957	0.957	0.950
ML Model Name	Observation about model performance
Logistic Regression	Performs well for linearly separable patterns. Provides stable precision and recall but may struggle with complex nonlinear relationships.
Decision Tree	Captures nonlinear relationships effectively but tends to overfit without pruning. Performance varies depending on tree depth.
kNN	Performs reasonably well after scaling. Sensitive to feature scaling and computationally expensive for large datasets.
Naive Bayes	Fast and computationally efficient. Performance slightly lower due to strong independence assumption between features.
Random Forest (Ensemble)	Strong and stable performance across all metrics. Reduces overfitting and captures complex feature interactions effectively.
XGBoost (Ensemble)	Achieved the best or near-best performance. High AUC, F1, and MCC due to boosting strategy and improved generalization ability.

