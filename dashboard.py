import streamlit as st
import pandas as pd
import numpy as np
import joblib # For loading the scaler
import json   # For loading the feature list
import tensorflow as tf
import os

# --- Configuration & Artifact Loading ---
# Define default paths (can be overridden if files are elsewhere)
# These should match the output names from your Colab pipeline
MODEL_FILENAME = "realtime_keras_model.keras"
SCALER_FILENAME = "realtime_feature_scaler.gz"
FEATURE_LIST_FILENAME = "realtime_feature_list.json"

# --- Helper Functions ---
@st.cache_resource # Cache the loaded model and scaler for performance
def load_artifacts(model_path, scaler_path, feature_list_path):
    """Loads the Keras model, scaler, and feature list."""
    loaded_model = None
    loaded_scaler = None
    expected_features = []

    errors = []

    if not os.path.exists(model_path):
        errors.append(f"Model file not found: {model_path}")
    else:
        try:
            loaded_model = tf.keras.models.load_model(model_path)
        except Exception as e:
            errors.append(f"Error loading Keras model: {e}")

    if not os.path.exists(scaler_path):
        errors.append(f"Scaler file not found: {scaler_path}")
    else:
        try:
            loaded_scaler = joblib.load(scaler_path)
        except Exception as e:
            errors.append(f"Error loading scaler: {e}")

    if not os.path.exists(feature_list_path):
        errors.append(f"Feature list file not found: {feature_list_path}")
    else:
        try:
            with open(feature_list_path, 'r') as f:
                expected_features = json.load(f)
            if not expected_features or not isinstance(expected_features, list):
                errors.append("Feature list is empty or not a valid list.")
                expected_features = []
        except Exception as e:
            errors.append(f"Error loading feature list: {e}")

    # Verification
    if loaded_model and expected_features:
        model_input_dim = loaded_model.input_shape[-1]
        if len(loaded_model.input_shape) == 3 and loaded_model.input_shape[-1] == 1:
            model_input_dim = loaded_model.input_shape[-2]
        if model_input_dim != len(expected_features):
            errors.append(f"Model input features ({model_input_dim}) != feature list ({len(expected_features)}).")

    return loaded_model, loaded_scaler, expected_features, errors

def get_port_category_streamlit(port_val: Optional[Any]) -> int: # Copied from app.py
    if port_val is None or pd.isna(port_val): return 0
    try:
        port = int(port_val)
        if 0 <= port <= 1023: return 1
        if 1024 <= port <= 49151: return 2
        if port >= 49152 : return 3
        return 0
    except (ValueError, TypeError):
        return 0

# --- Streamlit App UI ---
st.set_page_config(page_title="NIDS AI Model Tester", layout="wide")
st.title("🧪 Network Intrusion Detection - AI Model Tester")
st.markdown("""
This dashboard allows you to test the Keras model trained for detecting network anomalies.
Input feature values manually or upload a CSV with sample data.
The model was trained on features that can be approximated from packet-level data.
""")

# --- Load Artifacts ---
model, scaler, expected_features, load_errors = load_artifacts(
    MODEL_FILENAME, SCALER_FILENAME, FEATURE_LIST_FILENAME
)

if load_errors:
    for error in load_errors:
        st.error(error)
    st.warning("Please ensure the model, scaler, and feature list files are in the same directory as this script, or update the filenames at the top of the script.")
    st.stop() # Stop execution if artifacts can't be loaded

st.success(f"Model, scaler, and feature list ({len(expected_features)} features) loaded successfully!")
if model:
    st.subheader("Model Summary")
    # Capture model summary to string
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary_str = "\n".join(stringlist)
    st.text(model_summary_str)

# --- Input Method Selection ---
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose how to input data:", ("Manual Input", "Upload CSV"))

input_features_df = None

if input_method == "Manual Input":
    st.sidebar.subheader("Manual Feature Input")
    st.markdown("## Manual Feature Input")
    st.info(f"Enter values for the {len(expected_features)} expected features:")

    manual_inputs = {}
    cols = st.columns(3) # Adjust number of columns for layout
    col_idx = 0

    for feature_name in expected_features:
        # Provide sensible defaults or input types based on feature name
        if "Length" in feature_name or "Protocol" in feature_name:
            manual_inputs[feature_name] = cols[col_idx].number_input(f"{feature_name}", value=0.0, step=1.0, format="%.2f")
        elif "Is" in feature_name: # Boolean flags
            manual_inputs[feature_name] = float(cols[col_idx].checkbox(f"{feature_name} (1 if checked, 0 if not)", value=False))
        elif "Category" in feature_name:
            manual_inputs[feature_name] = float(cols[col_idx].selectbox(f"{feature_name}", options=[0,1,2,3], index=0))
        else: # Generic number input
            manual_inputs[feature_name] = cols[col_idx].number_input(f"{feature_name}", value=0.0, step=0.1, format="%.2f")
        col_idx = (col_idx + 1) % 3

    if st.button("Predict on Manual Input"):
        input_features_df = pd.DataFrame([manual_inputs])


elif input_method == "Upload CSV":
    st.sidebar.subheader("Upload CSV File")
    st.markdown("## Upload CSV for Batch Prediction")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Sample (first 5 rows):")
            st.dataframe(df_upload.head())

            # Check if all expected features are in the uploaded CSV
            missing_cols = [col for col in expected_features if col not in df_upload.columns]
            if missing_cols:
                st.error(f"The uploaded CSV is missing the following required columns: {', '.join(missing_cols)}")
                st.info(f"Expected columns are: {', '.join(expected_features)}")
            else:
                # Ensure correct order and select only expected features
                input_features_df = df_upload[expected_features].copy()
                st.success("CSV processed. Ready for prediction.")

        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")


# --- Prediction Logic ---
if input_features_df is not None and not input_features_df.empty:
    st.markdown("---")
    st.subheader("Prediction Results")

    try:
        # Ensure data types are float32 for scaling and model
        input_features_df = input_features_df.astype(np.float32)

        # Scale the features
        scaled_input_features = scaler.transform(input_features_df)

        # Reshape for model if necessary (e.g., for 1D CNN)
        model_input_s = model.input_shape
        reshaped_for_model: np.ndarray
        if len(model_input_s) == 3 and model_input_s[-1] == 1 and model_input_s[-2] == scaled_input_features.shape[1]:
            reshaped_for_model = scaled_input_features.reshape(scaled_input_features.shape[0], scaled_input_features.shape[1], 1)
            st.caption(f"Input reshaped to {reshaped_for_model.shape} for 1D CNN style model.")
        elif len(model_input_s) == 2 and model_input_s[-1] == scaled_input_features.shape[1]:
            reshaped_for_model = scaled_input_features
            st.caption(f"Input shape {reshaped_for_model.shape} for Dense style model.")
        else:
            st.error(f"Model input shape {model_input_s} incompatible with feature shape {scaled_input_features.shape}.")
            st.stop()


        # Make predictions
        predictions_proba = model.predict(reshaped_for_model)
        detection_threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.75, 0.01) # Allow user to adjust threshold

        results_list = []
        for i in range(len(predictions_proba)):
            proba = predictions_proba[i][0]
            is_anomaly = proba > detection_threshold
            label = "Anomaly" if is_anomaly else "Benign"
            results_list.append({
                "Sample": i + 1,
                "Probability": f"{proba:.4f}",
                "Prediction": label,
                "IsAnomaly": is_anomaly
            })

        results_df = pd.DataFrame(results_list)

        # Display results
        st.dataframe(results_df)

        # Summary statistics
        num_anomalies = results_df["IsAnomaly"].sum()
        num_total = len(results_df)
        st.metric(label="Anomalies Detected", value=f"{num_anomalies} / {num_total}")

        if num_anomalies > 0:
            st.warning(f"{num_anomalies} potential anomalies detected based on the threshold of {detection_threshold:.2f}.")
        else:
            st.success(f"No anomalies detected based on the threshold of {detection_threshold:.2f}.")

        # Show input features alongside predictions if it's a small batch
        if num_total <= 20: # Show details for small batches
            st.markdown("#### Detailed View (Input Features & Predictions)")
            detailed_df = input_features_df.reset_index(drop=True).join(
                results_df[["Probability", "Prediction"]].reset_index(drop=True)
            )
            st.dataframe(detailed_df)


    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        # import traceback
        # st.text(traceback.format_exc()) # For more detailed error in dashboard

else:
    if input_method == "Manual Input" and not st.button("Predict on Manual Input", key="initial_placeholder"): # Avoid re-triggering
        st.info("Enter feature values in the sidebar and click 'Predict on Manual Input'.")
    elif input_method == "Upload CSV" and uploaded_file is None:
        st.info("Upload a CSV file using the sidebar to make predictions.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for NIDS AI Model Testing.")
