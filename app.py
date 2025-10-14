# app.py
import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except Exception:
    GROQ_SDK_AVAILABLE = False

# dotenv to load env variables if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -----------------------
# Config / Globals
# -----------------------
st.set_page_config(page_title="Smart Factory Dashboard", layout="wide")

# Core model filenames
DEFECT_MODEL_FILE = "neu_defect_model.h5"
PM_MODEL_FILE = "xgb_predictive_maintenance.joblib"
SCALER_FILES = ["scaler.joblib", "pm_scaler.joblib"]
LSTM_MODEL_FILE = "lstm_model.h5"
FORECAST_SCALER_FILE = "scaler_forecast.joblib"
FORECAST_CSV = "monthly_retail_sales_cleaned.csv"  # now with month, Monthly_Sales

# optional background data
XSMOTE_FILE = "X_smote.csv"

DEFECT_CLASSES = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
REQUIRED_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Groq config (load from .env)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama3-70b-v2"

# -----------------------
# Utility helpers
# -----------------------
@st.cache_resource(show_spinner=False)
def load_defect_model(path=DEFECT_MODEL_FILE):
    if not os.path.exists(path):
        st.error(f"Defect model file not found: {path}")
        st.stop()
    return tf.keras.models.load_model(path)

@st.cache_resource(show_spinner=False)
def load_pm_model(path=PM_MODEL_FILE):
    if not os.path.exists(path):
        st.error(f"Predictive maintenance model not found: {path}")
        st.stop()
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_pm_scaler(choices=SCALER_FILES):
    for p in choices:
        if os.path.exists(p):
            return joblib.load(p)
    st.warning("No predictive-maintenance scaler found. Expected one of: " + ", ".join(choices))
    st.info("Create and save scaler.joblib or pm_scaler.joblib from training data, then restart the app.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_forecast_scaler(path=FORECAST_SCALER_FILE):
    if not os.path.exists(path):
        st.error(f"Forecast scaler not found: {path}")
        st.stop()
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_lstm_model(path=LSTM_MODEL_FILE):
    if not os.path.exists(path):
        st.error(f"LSTM forecast model not found: {path}")
        st.stop()
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"Failed to load LSTM model: {e}")
        st.stop()

def preprocess_image(image_file, target_size=(224,224)):
    image = Image.open(image_file).convert("RGB")
    image_resized = image.resize(target_size)
    arr = np.array(image_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, image

def format_feature_input(values_dict):
    arr = [values_dict[k] for k in REQUIRED_FEATURES]
    return np.array(arr, dtype=float).reshape(1, -1)

def log_to_csv(file_path, data_dict):
    df = pd.DataFrame([data_dict])
    if os.path.exists(file_path):
        old = pd.read_csv(file_path)
        pd.concat([old, df], ignore_index=True).to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)

# -----------------------
# Page layout
# -----------------------
module = st.sidebar.radio("Choose module", [
    "🏭 Manufacturer Dashboard",
    "🩻 Steel Defect Detection",
    "⚙️ Predictive Maintenance",
    "📈 Production Forecasting (LSTM)"
])

# -----------------------
# Manufacturer Dashboard
# -----------------------
# 🏭 Manufacturer Dashboard
if module == "🏭 Manufacturer Dashboard":
    # --- Page Header ---
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #f8f9fa 0%, #eef1f5 100%);
        }
        .stMetric {
            background: rgba(255, 255, 255, 0.7);
            border-radius: 12px;
            padding: 10px !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        }
        h1 {
            text-align: center;
            font-weight: 800;
            color: #333333;
        }
        .sub-title {
            text-align: center;
            color: #666;
            font-size: 18px;
            margin-bottom: 25px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>📊 Smart Steel Factory </h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Real-time overview of factory KPIs and performance efficiency</p>", unsafe_allow_html=True)

    # --- Top Metrics Row ---
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏭 Factories", "6 / 7")
    col2.metric("🧩 Production Lines", "26 / 32")
    col3.metric("⚙️ Machines", "120 / 130")
    col4.metric("👥 Users", "47 / 51")
    col5.metric("📦 Total Production", "260,000 Units")

    st.divider()
    st.markdown("### ⚙️ Equipment & Quality Overview")

    # --- Circular Gauges ---
    import plotly.graph_objects as go

    def circle_gauge(value, label, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={'suffix': '%'},
            title={'text': label, 'font': {'size': 20, 'color': '#333'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E0E0E0",
            }
        ))
        fig.update_layout(
            height=280,
            margin=dict(t=30, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': '#333', 'family': "Arial"}
        )
        return fig

    # --- Show Circular Charts ---
    colA, colB, colC = st.columns(3)
    colA.plotly_chart(circle_gauge(67.8, "OEE (Effectiveness)", "#FFB347"), use_container_width=True)
    colB.plotly_chart(circle_gauge(87.2, "Production Rate", "#36CFC9"), use_container_width=True)
    colC.plotly_chart(circle_gauge(99, "Quality Rate", "#9CDB4B"), use_container_width=True)

    # --- Footer Summary ---
    st.markdown("""
        <div style='text-align:center; color:#666; margin-top:25px; font-size:15px;'>
            💡 This dashboard visualizes key factory KPIs. Live data integration coming soon!
        </div>
    """, unsafe_allow_html=True)


    st.markdown("---")
    st.subheader("Defect Type Distribution (from logs)")
    defect_log = os.path.join(DATA_DIR, "defect_log.csv")
    if os.path.exists(defect_log):
        df_log = pd.read_csv(defect_log)
        chart_df = df_log["class"].value_counts().reset_index()
        chart_df.columns = ["Defect", "Count"]
        st.bar_chart(chart_df.set_index("Defect"))
    else:
        st.info("No defect data logged yet.")

    st.subheader("Predicted Failures (from logs)")
    maint_log = os.path.join(DATA_DIR, "maintenance_log.csv")
    if os.path.exists(maint_log):
        dfm = pd.read_csv(maint_log)
        st.write("Average Failure Probability: ", f"{dfm['failure_prob'].mean():.2f}%")
        st.line_chart(dfm["failure_prob"])
    else:
        st.info("No maintenance predictions logged yet.")

# -----------------------
# Steel defect detection
# -----------------------
elif module == "🩻 Steel Defect Detection":
    st.title("🩻 Steel Surface Defect Detection")
    uploaded_file = st.file_uploader("Upload steel image", type=["png","jpg","jpeg"])
    if uploaded_file:
        defect_model = load_defect_model()
        arr, pil_img = preprocess_image(uploaded_file)
        preds = defect_model.predict(arr)
        class_idx = int(np.argmax(preds, axis=1)[0])
        conf = float(np.max(preds))
        class_name = DEFECT_CLASSES[class_idx]
        st.image(pil_img, caption=f"Predicted: {class_name} ({conf*100:.2f}%)", use_container_width=True)
        probs_df = pd.DataFrame({"Class": DEFECT_CLASSES, "Probability": preds[0]})
        st.bar_chart(probs_df.set_index("Class"))

        log_to_csv(os.path.join(DATA_DIR, "defect_log.csv"), {
            "timestamp": pd.Timestamp.now(),
            "class": class_name,
            "confidence": conf
        })

# -----------------------
# Predictive Maintenance
# -----------------------
elif module == "⚙️ Predictive Maintenance":
    st.title("⚙️ Predictive Maintenance Checker")

    pm_model = load_pm_model()
    pm_scaler = load_pm_scaler()
    model_classes = getattr(pm_model, "classes_", None)

    # Single check
    st.subheader("🔧 Single Check")
    col1, col2 = st.columns(2)
    with col1:
        type_in = st.number_input("Machine Type (int)", min_value=0, max_value=20, value=2, step=1)
        air_temp = st.number_input("Air temperature [K]", min_value=0.0, max_value=1000.0, value=300.0)
        process_temp = st.number_input("Process temperature [K]", min_value=0.0, max_value=1000.0, value=310.0)
    with col2:
        rot_speed = st.number_input("Rotational speed [rpm]", min_value=0.0, max_value=10000.0, value=1200.0)
        torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=1000.0, value=35.0)
        tool_wear = st.number_input("Tool wear [min]", min_value=0.0, max_value=10000.0, value=150.0)

    if st.button("🔍 Predict Failure"):
        values = {
            "Type": float(type_in),
            "Air temperature [K]": float(air_temp),
            "Process temperature [K]": float(process_temp),
            "Rotational speed [rpm]": float(rot_speed),
            "Torque [Nm]": float(torque),
            "Tool wear [min]": float(tool_wear)
        }

        X = format_feature_input(values)
        Xs = pm_scaler.transform(X)

        pred = pm_model.predict(Xs)[0]
        probs = pm_model.predict_proba(Xs)[0]

        # robustly pick failure probability (class 1) if present
        if model_classes is not None:
            try:
                idx1 = int(np.where(model_classes == 1)[0][0])
                failure_prob = probs[idx1]
            except Exception:
                failure_prob = 1.0 - probs[0]
        else:
            failure_prob = probs[1] if len(probs) > 1 else (1.0 - probs[0])

        failure_pct = float(failure_prob * 100.0)

        if pred == 1:
            st.error(f"❌ FAILURE likely ({failure_pct:.4f}%)")
        else:
            st.success(f"✅ NORMAL operation ({failure_pct:.4f}%)")

        log_to_csv(os.path.join(DATA_DIR, "maintenance_log.csv"), {
            "timestamp": pd.Timestamp.now(),
            "failure_pred": int(pred),
            "failure_prob": failure_pct
        })

        # store for SHAP/Groq usage
        st.session_state["_last_input"] = X
        st.session_state["_last_input_scaled"] = Xs
        st.session_state["_last_pred"] = int(pred)
        st.session_state["_last_prob"] = float(failure_prob)

    # Simulation
    st.markdown("---")
    st.subheader("🧪 Simulation Panel – multiple scenarios")
    sim_count = st.slider("Number of scenarios to create", 1, 5, 2)
    scenario_list = []
    for i in range(sim_count):
        with st.expander(f"Scenario {i+1}", expanded=(i==0)):
            c1, c2, c3 = st.columns(3)
            with c1:
                s_type = st.number_input(f"Type (scenario {i+1})", min_value=0, max_value=20, value=2, key=f"type_{i}")
                s_air = st.slider(f"Air Temp (scenario {i+1})", 200.0, 500.0, 300.0, key=f"air_{i}")
                s_proc = st.slider(f"Process Temp (scenario {i+1})", 200.0, 600.0, 310.0, key=f"proc_{i}")
            with c2:
                s_speed = st.slider(f"Rot Speed (scenario {i+1})", 100.0, 5000.0, 1200.0, step=10.0, key=f"speed_{i}")
                s_torque = st.slider(f"Torque (scenario {i+1})", 0.0, 200.0, 35.0, key=f"torque_{i}")
            with c3:
                s_tool = st.slider(f"Tool wear (scenario {i+1})", 0.0, 1000.0, 150.0, key=f"tool_{i}")
            scenario_list.append({
                "Type": float(s_type),
                "Air temperature [K]": float(s_air),
                "Process temperature [K]": float(s_proc),
                "Rotational speed [rpm]": float(s_speed),
                "Torque [Nm]": float(s_torque),
                "Tool wear [min]": float(s_tool)
            })

    if st.button("Run Simulation Scenarios"):
        sim_results = []
        for i, sim in enumerate(scenario_list):
            Xs_sim = pm_scaler.transform(format_feature_input(sim))
            pred_sim = pm_model.predict(Xs_sim)[0]
            probs_sim = pm_model.predict_proba(Xs_sim)[0]
            if model_classes is not None:
                try:
                    idx1 = int(np.where(model_classes == 1)[0][0])
                    failure_sim = probs_sim[idx1]
                except Exception:
                    failure_sim = 1.0 - probs_sim[0]
            else:
                failure_sim = probs_sim[1] if len(probs_sim) > 1 else (1.0 - probs_sim[0])
            sim_results.append({
                "Scenario": i+1,
                "Input": sim,
                "Failure Prob (%)": round(float(failure_sim*100.0), 4),
                "Status": "FAILURE" if pred_sim == 1 else "NORMAL"
            })
        st.dataframe(pd.DataFrame(sim_results))

    # SHAP bar plot
    st.markdown("---")
    st.subheader("🔎 Feature impact (SHAP)")
    if not SHAP_AVAILABLE:
        st.warning("SHAP not installed. Install with `pip install shap` to enable explanations.")
    else:
        if "_last_input_scaled" not in st.session_state:
            st.info("Make a prediction first (Single Check) to enable SHAP explanation.")
        else:
            if not os.path.exists(XSMOTE_FILE):
                st.warning("X_smote.csv not found. SHAP will still run but a background sample would improve explanations.")
            if st.button("Show SHAP bar (feature impact)"):
                try:
                    explainer = shap.TreeExplainer(pm_model)
                    Xs_last = st.session_state["_last_input_scaled"]
                    shap_values = explainer.shap_values(Xs_last)
                except Exception as e:
                    st.error(f"SHAP explain failed: {e}")
                    shap_values = None

                if shap_values is not None:
                    if isinstance(shap_values, list):
                        try:
                            pred_idx = int(st.session_state["_last_pred"])
                            sv = np.abs(shap_values[pred_idx]).mean(axis=0)
                        except Exception:
                            sv = np.abs(shap_values[0]).mean(axis=0)
                    else:
                        sv = np.abs(shap_values).mean(axis=0)
                    feature_imp = pd.DataFrame({
                        "feature": REQUIRED_FEATURES,
                        "shap_abs_mean": sv.flatten()[:len(REQUIRED_FEATURES)]
                    }).sort_values("shap_abs_mean", ascending=False)
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.barh(feature_imp["feature"][::-1], feature_imp["shap_abs_mean"][::-1])
                    ax.set_xlabel("Mean |SHAP value| (higher = more impact)")
                    ax.set_title("Feature impact for last prediction")
                    st.pyplot(fig, bbox_inches='tight')

    # Groq chatbot for maintenance & model questions
    st.markdown("---")
    st.subheader("🤖 Ask the model & dataset (Groq chatbot)")
    if not GROQ_SDK_AVAILABLE:
        st.warning("Groq SDK not installed. Install it to enable the chatbot.")
    elif not GROQ_API_KEY:
        st.warning("GROQ_API_KEY not found in environment (.env). Add it then restart the app.")
    else:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
        except Exception as e:
            st.error(f"Failed to create Groq client: {e}")
            groq_client = None

        user_question = st.text_input("Ask a question about the model or dataset (e.g. 'Why is failure prob low?')")
        if st.button("Ask Groq") and groq_client is not None:
            last_input = st.session_state.get("_last_input", None)
            last_pred = st.session_state.get("_last_pred", None)
            last_prob = st.session_state.get("_last_prob", None)
            background_info = ""
            if os.path.exists(XSMOTE_FILE):
                try:
                    df_bg = pd.read_csv(XSMOTE_FILE)
                    background_info = (f"Training set samples: {len(df_bg)}, features: {list(df_bg.columns[:len(REQUIRED_FEATURES)])}.")
                except Exception:
                    background_info = "Background data available but failed to read."
            prompt = (
                "You are an expert data scientist. Answer briefly and clearly.\n\n"
                f"{background_info}\n"
                f"Recent prediction details:\n- Input: {last_input.tolist() if last_input is not None else 'N/A'}\n"
                f"- Predicted class: {last_pred}\n- Failure probability: {last_prob}\n\n"
                "User question:\n" + user_question + "\n\n"
                "Explain which features are contributing most and why, and suggest any data/model checks."
            )
            try:
                response = groq_client.chat.completions.create(
                    model=GROQ_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400
                )
                answer = response.choices[0].message["content"]
                st.info(answer)
            except Exception as e:
                st.error(f"Groq call failed: {e}")


# -----------------------
# LSTM Forecasting
# -----------------------
elif module == "📈 Production Forecasting (LSTM)":
    st.title("📈 Future Production Forecast – LSTM Model")

    # File checks
    if not os.path.exists(LSTM_MODEL_FILE):
        st.error(f"LSTM model file not found: {LSTM_MODEL_FILE}")
        st.stop()
    if not os.path.exists(FORECAST_SCALER_FILE):
        st.error(f"Forecast scaler not found: {FORECAST_SCALER_FILE}")
        st.stop()
    if not os.path.exists(FORECAST_CSV):
        st.error(f"Forecast CSV not found: {FORECAST_CSV}")
        st.stop()

    # Load LSTM + Scaler
    lstm_model = load_lstm_model(LSTM_MODEL_FILE)
    forecast_scaler = load_forecast_scaler(FORECAST_SCALER_FILE)

    # ✅ FIXED: Correct CSV column handling
    df_hist = pd.read_csv(FORECAST_CSV)
    df_hist = df_hist.rename(columns={"month": "Date", "Monthly_Sales": "Production"})
    df_hist["Date"] = pd.to_datetime(df_hist["Date"], errors="coerce")

    if "Production" not in df_hist.columns:
        st.error("CSV must contain a 'Production' column.")
        st.stop()

    df_hist = df_hist.sort_values("Date").reset_index(drop=True)
    series = df_hist.set_index("Date")["Production"].astype(float)

    st.subheader("📊 Historical Production Data")
    st.line_chart(series)

    # Forecast parameters
    st.subheader("Forecast settings")
    seq_len = st.number_input("Sequence length used during training (timesteps)", min_value=1, max_value=60, value=12)
    months = st.slider("Months ahead to forecast", 1, 24, 12)

    if len(series) < seq_len:
        st.error(f"Not enough history for seq_len={seq_len}. Need at least {seq_len} months.")
    else:
        last_vals = series.values[-seq_len:].reshape(-1, 1)
        scaled_seq = forecast_scaler.transform(last_vals).reshape(1, seq_len, 1)

        preds_scaled = []
        seq = scaled_seq.copy()
        for _ in range(months):
            pred_scaled = lstm_model.predict(seq, verbose=0)[0][0]
            preds_scaled.append(pred_scaled)
            seq = np.append(seq[:, 1:, :], [[[pred_scaled]]], axis=1)

        preds_rescaled = forecast_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

        last_date = series.index[-1]
        future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=months, freq="MS")
        forecast_df = pd.DataFrame({"Forecasted Production": preds_rescaled}, index=future_dates)

        st.subheader("📉 Forecasted Production")
        combined = pd.concat([series, forecast_df["Forecasted Production"]], axis=1)
        combined.columns = ["Historical", "Forecast"]
        st.line_chart(combined.fillna(method="ffill"))

        st.write("Forecast table (next months):")
        st.dataframe(forecast_df)

        if st.button("Save forecast to CSV"):
            out_file = "future_12_months_forecast_table.csv"
            forecast_df.to_csv(out_file)
            st.success(f"Saved forecast to {out_file}")

st.markdown("---")
st.caption("Smart Factory Dashboard • CNN Defect Detection + Predictive Maintenance + LSTM Forecasting • Streamlit")
