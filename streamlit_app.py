import streamlit as st
import pandas as pd
import joblib


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
    "Enter your data on the left and click **Predict** to see your obesity level."
)

st.sidebar.header("Your Health details")
# -----------------------------
# Inputs
# -----------------------------
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender_f = 1 if gender == "Female" else 0
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
height = st.sidebar.number_input("Height", min_value=0.0, max_value=7.0, value=5.5)
weight = st.sidebar.number_input("Weight", min_value=0.0, max_value=500.0, value=100.0)
family_history_with_overweight = st.sidebar.selectbox("Family History with Overweight", ["Yes", "No"])
family_history_with_overweight_y = 1 if family_history_with_overweight == "Yes" else 0
favc = st.sidebar.selectbox("Frequent consumption of high-calorie food", ["Yes", "No"])
favc_y = 1 if favc == "Yes" else 0
fcvc = st.sidebar.number_input("Frequency of consumption of vegetables", min_value=0, max_value=3, value=1)
ncp = st.sidebar.number_input("Number of main meals", min_value=0, max_value=5, value=2)
caec = st.sidebar.selectbox("Consumption of alcohol", ["Yes", "No"])
caec_y = 1 if caec == "Yes" else 0
smoke = st.sidebar.selectbox("Smoking", ["Yes", "No"])
smoke_y = 1 if smoke == "Yes" else 0
ch2o = st.sidebar.number_input("Consumption of water per day", min_value=0.0, max_value=10.0, value=2.0)
scc = st.sidebar.selectbox("Consumption of fruit per day", ["Yes", "No"])
scc_y = 1 if scc == "Yes" else 0
faf = st.sidebar.number_input("Frequency of consumption of fast food", min_value=0, max_value=3, value=1)
tue = st.sidebar.number_input("Consumption of vegetables per day", min_value=0, max_value=3, value=1)
calc = st.sidebar.selectbox("Consumption of alcohol per day", ["Yes", "No"])
calc_y = 1 if calc == "Yes" else 0
mtrans = st.sidebar.selectbox("Transportation used", ["Automobile", "Motorbike", "Bike", "Public Transportation", "Walking"])
mtrans_value = 0 if mtrans == "Automobile" else 1 if mtrans == "Motorbike" else 2 if mtrans == "Bike" else 3 if mtrans == "Public Transportation" else 4


data_in = {
    FEATURE_COLUMNS[0]: gender_f,
    FEATURE_COLUMNS[1]: age,
    FEATURE_COLUMNS[2]: height,
    FEATURE_COLUMNS[3]: weight,
    FEATURE_COLUMNS[4]: family_history_with_overweight_y,
    FEATURE_COLUMNS[5]: favc_y,
    FEATURE_COLUMNS[6]: fcvc,
    FEATURE_COLUMNS[7]: ncp,
    FEATURE_COLUMNS[8]: caec_y,
    FEATURE_COLUMNS[9]: smoke_y,
    FEATURE_COLUMNS[10]: ch2o,
    FEATURE_COLUMNS[11]: scc_y,
    FEATURE_COLUMNS[12]: faf,
    FEATURE_COLUMNS[13]: tue,
    FEATURE_COLUMNS[14]: calc_y,
    FEATURE_COLUMNS[15]: mtrans_value
}

data = pd.DataFrame([data_in], columns=FEATURE_COLUMNS)

#uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])

#if uploaded_file is not None:
    #data = pd.read_csv(uploaded_file)
data_processed = pipeline.transform(data)

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
    
        st.write("Predictions:")
        st.success(f"Predicted Obesity Level: {predicted_class, class_labels[predicted_class]}")

        st.write("### Input Features Sent to Model")
        st.dataframe(data)
else:
    st.info("Fill the details on the left and click **Predict my obesity level**.")
