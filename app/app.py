from pathlib import Path
import streamlit as st
import joblib
import pandas as pd

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Network Intrusion Detection", page_icon="🛡️")

st.title("🛡️ Network Intrusion Detection")
st.write("App loaded successfully.")

# --- DYNAMIC FILE PATH SETUP ---
# BASE_DIR targets 'D:\Network_intrusion_detection\app'
BASE_DIR = Path(__file__).resolve().parent

# 🛠️ FIX: Added '.parent' to climb out of 'app' and into the root folder 
# This correctly matches your layout: D:\Network_intrusion_detection\models\...
model_path = BASE_DIR.parent / "models" / "network_intrusion_model.pkl"

# --- MACHINE LEARNING MODEL LOADING ---
if model_path.exists():
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
    
    # Save how many features the model expects for validation checks later
    n_features = model.n_features_in_
    st.write(f"Model expects **{n_features}** features")
else:
    # Safely halt the app if the model file is missing from GitHub or the directory
    st.error(f"❌ Model file not found at: `{model_path}`. Please verify you have pushed it to GitHub.")
    st.stop() 

# --- SIDEBAR & OPTIONAL DATA VERIFICATION ---
st.sidebar.title("Options")
show_training_data = st.sidebar.checkbox("🧪 Show Training Data", value=False)

if show_training_data:
    st.subheader("🧪 Training Data")
    # 🛠️ FIX: Added '.parent' here as well to reach D:\Network_intrusion_detection\data\...
    data_path = BASE_DIR.parent / "data" / "Train_data.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Target:** {df['class'].value_counts().to_dict()}")
        st.dataframe(df.head())
    else:
        st.warning(f"⚠️ Train_data.csv not found at `{data_path}`")

# --- INTERACTIVE FILE UPLOADER FOR PREDICTIONS ---
st.subheader("🔍 Upload CSV File for Intrusion Detection")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Reset file pointer to the beginning of the file to ensure safe reading
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    
    st.write(f"📄 Uploaded: {uploaded_file.name}")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())

    # --- INFERENCE TRIGGER BUTTON ---
    if st.button("🚀 Predict Intrusion"):
        try:
            # 1. Feature Selection: Drop columns not utilized during training phase
            df = df.drop(columns=['num_outbound_cmds', 'is_host_login', 'class'], errors='ignore')

            # 2. Categorical Variable Encoding: Map string features to identical numeric tokens used in training
            if 'protocol_type' in df.columns:
                df['protocol_type'] = df['protocol_type'].map({'tcp': 0, 'udp': 1, 'icmp': 2})
            if 'service' in df.columns:
                df['service'] = df['service'].map({'http': 0, 'ftp': 1, 'smtp': 2, 'telnet': 3, 'other': 4})
            if 'flag' in df.columns:
                df['flag'] = df['flag'].map({'SF': 0, 'S0': 1, 'S1': 2, 'S2': 3})

            # 3. Shape Validation: Ensure current feature matrix matches what the model expects
            if df.shape[1] != n_features:
                st.error(f"❌ Feature mismatch! Expected {n_features}, got {df.shape[1]}")
            else:
                # 4. Generate Predictions
                prediction = model.predict(df)
                st.success("✅ Predictions generated!")
                st.write(f"**Count:** {len(prediction)}")

                # 5. Summary Analysis: Display breakdown of normal vs anomaly results
                unique_preds = pd.Series(prediction).value_counts()
                st.subheader("Prediction Summary:")
                st.dataframe(unique_preds)

                # 6. Append Results: Join predictions back into the visible table
                df['prediction'] = prediction
                st.subheader("Data with Predictions:")
                st.dataframe(df.head())

                # 7. File Compilation: Convert updated DataFrame into downloadable format
                csv_output = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_output,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Prediction error: {e}")

# --- MODEL METADATA METRICS ---
st.subheader("📊 Model Information")
st.write(f"Model type: {model.__class__.__name__}")