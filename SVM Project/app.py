import streamlit as st 
import pandas as pd 
import numpy as np 
import os
import seaborn as sns 
import matplotlib.pyplot as plt 
import requests 
from datetime import datetime

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# logger
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Session state intitialization    
if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved = False

# Folder setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

log("Application Started")
log(f"Raw Directory: {RAW_DIR}")
log(f"Cleaned Directory: {CLEAN_DIR}")

# Page Configuration
st.set_page_config("End to End SVM", layout = 'wide')
st.title("End-to-end SVM Platform")

# Side bar : model settings
st.sidebar.header("SVM Settings")
kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

log(f"SVM Settings -> kernel = {kernel}, C = {C}, Gamma = {gamma}")

# Step 1 : Data Ingestion

st.header("Step 1: Data Ingestion")
log("Step 1 started: Data Ingestion")

option = st.radio("Choose Data source ", ["Download Dataset", "Upload CSV"])

df = None
raw_path = None

if option == "Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris Dataset")
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)

        if response.status_code != 200:
            st.error("Failed to download dataset")
            st.stop()

        raw_path = os.path.join(RAW_DIR, "iris.csv")

        with open(raw_path, "wb") as f:
            f.write(response.content)

        if os.path.getsize(raw_path) == 0:
            st.error("Downloaded file is empty")
            st.stop()

        df = pd.read_csv(raw_path)
        st.success("Dataset downloaded successfully!")
        log(f"Iris dataset saved at {raw_path}")

if option == "Upload CSV":
    upload_file = st.file_uploader("Upload CSV file", type=["csv"])
    if upload_file:
        raw_path = os.path.join(RAW_DIR, upload_file.name)

        with open(raw_path, "wb") as f:
            f.write(upload_file.getbuffer())

        if os.path.getsize(raw_path) == 0:
            st.error("Uploaded file is empty")
            st.stop()

        df = pd.read_csv(raw_path)
        st.success("File uploaded successfully!")
        log(f"Uploaded dataset saved at {raw_path}")

# Step 2 : EDA
if df is not None:
    st.header("Step 2 : EDA")
    log("Step 2 started: EDA")

    st.dataframe(df.head())
    st.write("Shape: ", df.shape)
    st.write("Missing values: ", df.isnull().sum())
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only = True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    log("EDA Completed")

# Step 3 : Data Cleaning
if df is not None:
    st.header("Step 3 : Data Cleaning")
    log("Step 3 started: Data Cleaning")
    strategy = st.selectbox(
        "Missing value strategy",
        ["Mean", "Median", "Drop Rows"]
    )

    df_clean = df.copy()
    if strategy == "Drop Rows":
        df_clean = df_clean.dropna()
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy == "Mean":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        st.session_state.df_clean = df_clean
        st.success("Data Cleaning Completed!")

else:
    st.info("Please complete Step 1 (data Ingestion) first...")

# Step 4 : Save Cleaned Data
if st.button("Save cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("No Cleaned data found. Please complete Step 3 first...")
    else : 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = f"cleaned_dataset_{timestamp}.csv"
        clean_path = os.path.join(CLEAN_DIR, clean_filename)

        st.session_state.df_clean.to_csv(clean_path, index=False)
        st.success("Cleaned Dataset Saved!")
        st.info(f"Saved at : {clean_path}")
        log(f"Cleaned dataset saved at : {clean_path}")

# Step 5 : Load Cleaned Dataset
st.header("Step 5 : Load Cleaned Dataset")
clean_file = os.listdir(CLEAN_DIR)
if not clean_file:
    st.warning("No cleaned datasets found. Please save one in Step 4...")
    log("No cleaned dataset available in the terminal")

else:
    selected = st.selectbox("Select Cleaned Dataset:", clean_file)
    df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected))

    st.success(f"Loaded dataset: {selected}")
    log(f"Loaded Cleaned dataset: {selected}")
    st.dataframe(df_model.head())

# Step 6 : Train SVM
st.header("Step 6 : Train SVM")
log("Step 6 started : Train SVM")

# Restrict target to categorical columns only
categorical_cols = df_model.select_dtypes(include=["object", "category"]).columns

if len(categorical_cols) == 0:
    st.error("No categorical columns available for SVM classification.")
    st.stop()

target = st.selectbox("Select Target Column", categorical_cols)

# Encode target
le = LabelEncoder()
y = le.fit_transform(df_model[target])
log("Target column encoded")

# Select numerical features only
x = df_model.drop(columns=[target])
x = x.select_dtypes(include=np.number)

if x.empty:
    st.error("No numeric features available for training.")
    st.stop()

# Scale features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# Train SVM
model = SVC(kernel=kernel, C=C, gamma=gamma)
model.fit(x_train, y_train)

# Evaluation
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Accuracy : {acc:.2f}")
log(f"SVM Trained Successfully | Accuracy = {acc:.2f}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)