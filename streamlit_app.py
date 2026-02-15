import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Obesity Level Classification App")

# Load models
lr = joblib.load("logistic_regression_model.pkl")
dt = joblib.load("decision_tree_model.pkl")
knn = joblib.load("knn_model.pkl")
nb = joblib.load("naive_bayes_model.pkl")
rf = joblib.load("random_forest_model.pkl")
xgb = joblib.load("xgboost_model.pkl")
pipeline = joblib.load("preprocessing_pipeline.pkl")

model_option = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

FEATURE_COLUMNS = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
       'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
       'CALC', 'MTRANS']
class_labels = {
        0: "Insufficient Weight",
        1: "Normal Weight",
        2: "Obesity Type I",
        3: "Obesity Type II",
        4: "Obesity Type III",
        5: "Overweight Level I",
        6: "Overweight Level II"
}

st.write(
    "Enter your data on the left in CSV format or enter separately and click **Predict** to see your obesity level."
)

#st.sidebar.header("Your Health details")

uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Test Data")
    st.dataframe(test_data.head())
    
    X_test = test_data.drop("NObeyesdad", axis=1)
    y_test = test_data["NObeyesdad"]

    data_processed = pipeline.transform(X_test)
    st.write("Columns in uploaded file:", X_test.columns)
    st.write("Columns expected by pipeline:", pipeline.feature_names_in_)

    if st.button("Predict my obesity level"):

        if model_option == "Logistic Regression":
                prediction = lr.predict(data_processed)
        elif model_option == "Decision Tree":
                prediction = dt.predict(data_processed)
        elif model_option == "KNN":
                prediction = knn.predict(data_processed)
        elif model_option == "Naive Bayes":
                prediction = nb.predict(data_processed)
        elif model_option == "Random Forest":
                prediction = rf.predict(data_processed)
        else:
                prediction = xgb.predict(data_processed)

        
        predicted_class = prediction[0]
        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction, average='weighted')
        recall = recall_score(y_test, prediction, average='weighted')
        f1 = f1_score(y_test, prediction, average='weighted')
        auc = roc_auc_score(y_test, model.predict_proba(data_processed), multi_class='ovr')
        mcc = matthews_corrcoef(y_test, prediction)
    
        st.write("Predictions:")
        st.success(f"Predicted Obesity Level: {predicted_class, class_labels[predicted_class]}")

        st.write("## Confusion Matrix")

        cm = confusion_matrix(y_test, prediction)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

        st.write("## Classification Report")

        report = classification_report(y_test, y_pred)
        st.text(report)

        st.write("## Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")

        col1.metric("F1 Score", f"{f1:.4f}")
        col2.metric("MCC", f"{mcc:.4f}")
        col3.metric("AUC", auc)

        st.write("### Input Features Sent to Model")
        st.dataframe(data)
else:
    st.info("Fill the details on the left and click **Predict my obesity level**.")
