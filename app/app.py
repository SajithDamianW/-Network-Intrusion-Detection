from pathlib import Path
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Network Intrusion Detection", page_icon="🛡️")

st.title("🛡️ Network Intrusion Detection")
st.write("App loaded successfully.")

# Load model
model_path = Path(r"D:\Network_intrusion_detection\models\network_intrusion_model.pkl")
model = joblib.load(model_path)
st.success("✅ Model loaded successfully!")

n_features = model.n_features_in_
st.write(f"Model expects **{n_features}** features")

# Sidebar
st.sidebar.title("Options")
show_training_data = st.sidebar.checkbox("🧪 Show Training Data", value=False)

# Training data (optional)
if show_training_data:
    st.subheader("🧪 Training Data")
    data_path = Path(r"D:\Network_intrusion_detection\data\Train_data.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Target:** {df['class'].value_counts().to_dict()}")
        st.dataframe(df.head())
    else:
        st.warning("⚠️ Train_data.csv not found")

# Upload CSV for prediction
st.subheader("🔍 Upload CSV File for Intrusion Detection")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    st.write(f"📄 Uploaded: {uploaded_file.name}")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())

    if st.button("🚀 Predict Intrusion"):
        try:
            # ✅ Preprocessing — must match notebook exactly
            df = df.drop(columns=['num_outbound_cmds', 'is_host_login', 'class'], errors='ignore')

            if 'protocol_type' in df.columns:
                df['protocol_type'] = df['protocol_type'].map({'tcp': 0, 'udp': 1, 'icmp': 2})
            if 'service' in df.columns:
                df['service'] = df['service'].map({'http': 0, 'ftp': 1, 'smtp': 2, 'telnet': 3, 'other': 4})
            if 'flag' in df.columns:
                df['flag'] = df['flag'].map({'SF': 0, 'S0': 1, 'S1': 2, 'S2': 3})

            if df.shape[1] != n_features:
                st.error(f"❌ Feature mismatch! Expected {n_features}, got {df.shape[1]}")
            else:
                prediction = model.predict(df)
                st.success("✅ Predictions generated!")
                st.write(f"**Count:** {len(prediction)}")

                unique_preds = pd.Series(prediction).value_counts()
                st.subheader("Prediction Summary:")
                st.dataframe(unique_preds)

                df['prediction'] = prediction
                st.subheader("Data with Predictions:")
                st.dataframe(df.head())

                csv_output = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_output,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.subheader("📊 Model Information")
st.write(f"Model type: {model.__class__.__name__}")