 # app.py
import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from groq import Groq
from supabase import create_client


# optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Groq replacement for Gemini
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
DEFECT_MODEL_FILE = "textile_model (2).h5"
PM_MODEL_FILE = "xgb_predictive_maintenance.joblib"
SCALER_FILES = ["scaler.joblib", "pm_scaler.joblib"]
LSTM_MODEL_FILE = "lstm_model.h5"
FORECAST_SCALER_FILE = "scaler_forecast.joblib"
FORECAST_CSV = "monthly_retail_sales_cleaned.csv"  # now with month, Monthly_Sales

# optional background data
XSMOTE_FILE = "X_smote.csv"

DEFECT_CLASSES = [
    'Broken stitch', 
    'Needle mark', 
    'Pinched fabric', 
    'Vertical', 
    'defect free', 
    'hole', 
    'horizontal', 
    'lines', 
    'stain'
]
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

# Groq config (Replacing Gemini)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_AVAILABLE = GROQ_API_KEY is not None

if GROQ_AVAILABLE:
    client = Groq(api_key=GROQ_API_KEY)
# -----------------------
# Supabase Config
# -----------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SUPABASE_AVAILABLE = SUPABASE_URL and SUPABASE_KEY

if SUPABASE_AVAILABLE:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None



# -----------------------
# Utility helpers
# -----------------------
@st.cache_resource(show_spinner=False, ttl=0)
def load_defect_model(path=DEFECT_MODEL_FILE):
    if not os.path.exists(path):
        st.error(f"Model file nahi mili: {path}")
        st.stop()
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        st.error(f"Model load karne mein error: {e}")
        st.stop()

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

def fetch_latest_machine_data():
    if supabase is None:
        return None

    response = (
        supabase
        .table("machine_telemetry")
        .select("*")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    row = response.data[0]

    return {
        "Type": float(row["type"]),
        "Air temperature [K]": float(row["air_temp"]),
        "Process temperature [K]": float(row["process_temp"]),
        "Rotational speed [rpm]": float(row["rpm"]),
        "Torque [Nm]": float(row["torque"]),
        "Tool wear [min]": float(row["tool_wear"])
    }


def encode_image(image_pil):
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_defect_with_groq(image_pil, defect_class):
    try:
        base64_image = encode_image(image_pil)
        prompt = f"""
        You are a textile manufacturing quality control expert.
        The vision model predicted this textile defect class: {defect_class}.
        Analyze the uploaded fabric surface image and provide:
        1. Typical visual markers for {defect_class}.
        2. Severity level (Low / Medium / High)
        3. Root cause (e.g., needle tension, loom synchronization, or yarn contamination).
        4. Corrective and Preventive actions for the production line.
        6. Preventive measures to avoid recurrence
        """
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq analysis failed: {e}"


# -----------------------
# Page layout
# -----------------------
from urllib.parse import quote

MENU = [
    ("manufacturer", "🏭 Home"),
    ("defect", "🩻 Textile Defect Detection"),
    ("pm", "⚙️ Predictive Maintenance"),
    ("forecast", "📈 Production Forecasting (LSTM)")
]

_qparams = st.query_params
_qmodule = _qparams.get("module", "manufacturer")

if isinstance(_qmodule, list):
    active_key = _qmodule[0] if len(_qmodule) > 0 else "manufacturer"
elif _qmodule is None or _qmodule == "":
    active_key = "manufacturer"
else:
    active_key = str(_qmodule)

st.sidebar.markdown(
    """
    <style>
    .sidebar-menu { font-family: "Inter", sans-serif; padding: 8px 8px 20px 8px; }
    .menu-title { color: #f1f3f5; font-size: 13px; margin: 8px 12px; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.06em; }
    .menu-item { display: flex; align-items: center; gap: 10px; padding: 10px 12px; margin: 6px 6px; border-radius: 10px; text-decoration: none; color: #dde3e8; background: transparent; transition: background 0.12s ease, transform 0.06s ease; border: 1px solid transparent; }
    .menu-item:hover { background: rgba(255,255,255,0.02); transform: translateY(-1px); color: #fff; }
    .menu-item .icon { font-size: 18px; width: 26px; text-align: center; }
    .menu-item .label { font-weight: 600; font-size: 15px; }
    .menu-item.active { background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border: 1px solid rgba(255,255,255,0.04); box-shadow: 0 6px 18px rgba(0,0,0,0.45) inset; color: #ffffff; }
    .menu-footer { color: #9aa3ab; font-size: 12px; padding: 8px 12px 20px 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

menu_html = "<div class='sidebar-menu'><div class='menu-title'></div>"
for key, label in MENU:
    icon = label.split()[0]
    text = " ".join(label.split()[1:])
    active_cls = "active" if key == active_key else ""
    href = f"?module={quote(key)}"
    menu_html += (
        f"<a class='menu-item {active_cls}' href='{href}'>"
        f"<div class='icon'>{icon}</div>"
        f"<div class='label'>{text}</div>"
        f"</a>"
    )
menu_html += "<div class='menu-footer'>Quick links: dashboard • defect detector • maintenance • forecasting</div></div>"
st.sidebar.markdown(menu_html, unsafe_allow_html=True)

_map = {
    "manufacturer": "🏭 Manufacturer Dashboard",
    "defect": "🩻 Textile Defect Detection",
    "pm": "⚙️ Predictive Maintenance",
    "forecast": "📈 Production Forecasting (LSTM)"
}
module = _map.get(active_key, "🏭 Manufacturer Dashboard")

# -----------------------
# Manufacturer Dashboard
# -----------------------
if module == "🏭 Manufacturer Dashboard":
    st.markdown("""
        <style>
        .main { background: linear-gradient(135deg, #f8f9fa 0%, #eef1f5 100%); }
        .stMetric { background: rgba(255, 255, 255, 0.7); border-radius: 12px; padding: 10px !important; box-shadow: 0 4px 8px rgba(0,0,0,0.08); }
        h1 { text-align: center; font-weight: 800; color: #333333; }
        .sub-title { text-align: center; color: #666; font-size: 18px; margin-bottom: 25px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>📊 FactoryMindAI </h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Real-time overview of factory KPIs and performance efficiency</p>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏭 Factories", "6 / 7")
    col2.metric("🧩 Production Lines", "26 / 32")
    col3.metric("⚙️ Machines", "120 / 130")
    col4.metric("👥 Users", "47 / 51")
    col5.metric("📦 Total Production", "260,000 Units")

    st.divider()
    st.markdown("### ⚙️ Equipment & Quality Overview")

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
        fig.update_layout(height=280, margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", font={'color': '#333'})
        return fig

    colA, colB, colC = st.columns(3)
    colA.plotly_chart(circle_gauge(67.8, "OEE (Effectiveness)", "#FFB347"), use_container_width=True)
    colB.plotly_chart(circle_gauge(87.2, "Production Rate", "#36CFC9"), use_container_width=True)
    colC.plotly_chart(circle_gauge(99, "Quality Rate", "#9CDB4B"), use_container_width=True)

    st.markdown("---")
    st.subheader("Defect Type Distribution (from logs)")
    defect_log = os.path.join(DATA_DIR, "defect_log.csv")
    if os.path.exists(defect_log):
        df_log = pd.read_csv(defect_log)
        chart_df = df_log["class"].value_counts().reset_index()
        chart_df.columns = ["Defect", "Count"]
        st.bar_chart(chart_df.set_index("Defect"))

    st.subheader("Predicted Failures (from logs)")
    maint_log = os.path.join(DATA_DIR, "maintenance_log.csv")
    if os.path.exists(maint_log):
        dfm = pd.read_csv(maint_log)
        st.write("Average Failure Probability: ", f"{dfm['failure_prob'].mean():.2f}%")
        st.line_chart(dfm["failure_prob"])

# -----------------------
# Textile defect detection
# -----------------------
elif module == "🩻 Textile Defect Detection":
    st.title("🩻 Textile Surface Defect Detection")
    uploaded_file = st.file_uploader("Upload textile image", type=["png","jpg","jpeg"])
    
    if uploaded_file:
        defect_model = load_defect_model()
        arr, pil_img = preprocess_image(uploaded_file)
        preds = defect_model.predict(arr)
        class_idx = int(np.argmax(preds, axis=1)[0])
        conf = float(np.max(preds))
        class_name = DEFECT_CLASSES[class_idx]
        
        # UI Change: Two columns to show image and analysis side-by-side
        col_img, col_analysis = st.columns([1, 1.2]) 

        with col_img:
            # Resize image to be smaller in the UI
            st.image(pil_img, caption=f"Predicted: {class_name} ({conf*100:.2f}%)", use_container_width=True)
            probs_df = pd.DataFrame({"Class": DEFECT_CLASSES, "Probability": preds[0]})
            st.bar_chart(probs_df.set_index("Class"))

        with col_analysis:
            st.markdown("### 🧠 AI Textile Defect Expert Analysis")
            if not GROQ_AVAILABLE:
                st.warning("Groq API key not configured.")
            else:
                with st.spinner("Analyzing textile defect using Groq Llama Vision..."):
                    groq_report = analyze_defect_with_groq(pil_img, class_name)
                    st.success("Analysis Complete")
                    st.markdown(groq_report)

        log_to_csv(os.path.join(DATA_DIR, "defect_log.csv"), {
            "timestamp": pd.Timestamp.now(),
            "class": class_name,
            "confidence": conf
        })

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
    
    st.markdown("---")
    st.subheader("📡 Live Machine Data (Supabase Auto Fetch)")
    
    machine_data = None
    if not SUPABASE_AVAILABLE:
        st.warning("Supabase not configured. Add SUPABASE_URL and SUPABASE_KEY to .env")
    else:
        if st.button("Fetch Latest Supabase Data & Predict"):
            machine_data = fetch_latest_machine_data()

        if machine_data is None:
            st.warning("No machine data found in Supabase.")
        else:
            st.write("📥 Latest Telemetry", machine_data)

            X = format_feature_input(machine_data)
            Xs = pm_scaler.transform(X)

            pred = pm_model.predict(Xs)[0]
            probs = pm_model.predict_proba(Xs)[0]

            if model_classes is not None:
                try:
                    idx1 = int(np.where(model_classes == 1)[0][0])
                    failure_prob = probs[idx1]
                except Exception:
                    failure_prob = 1.0 - probs[0]
            else:
                failure_prob = probs[1] if len(probs) > 1 else (1.0 - probs[0])

            failure_pct = float(failure_prob * 100)

            if pred == 1:
                st.error(f"❌ FAILURE Likely ({failure_pct:.2f}%)")
            else:
                st.success(f"✅ NORMAL Operation ({failure_pct:.2f}%)")

            log_to_csv(os.path.join(DATA_DIR, "maintenance_log.csv"), {
                "timestamp": pd.Timestamp.now(),
                **machine_data,
                "failure_pred": int(pred),
                "failure_prob": failure_pct
            })


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
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
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

    if not os.path.exists(LSTM_MODEL_FILE) or not os.path.exists(FORECAST_SCALER_FILE):
        st.error("Missing model or scaler files.")
        st.stop()

    lstm_model = load_lstm_model(LSTM_MODEL_FILE)
    forecast_scaler = load_forecast_scaler(FORECAST_SCALER_FILE)

    df_hist = pd.read_csv(FORECAST_CSV)
    df_hist = df_hist.rename(columns={"month": "Date", "Monthly_Sales": "Production"})
    df_hist["Date"] = pd.to_datetime(df_hist["Date"], errors="coerce")
    df_hist = df_hist.sort_values("Date").reset_index(drop=True)
    series = df_hist.set_index("Date")["Production"].astype(float)

    st.subheader("📊 Historical Production Data")
    st.line_chart(series)

    seq_len = st.number_input("Sequence length", min_value=1, max_value=60, value=12)
    months = st.slider("Months ahead", 1, 24, 12)

    if len(series) >= seq_len:
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
        st.line_chart(combined.ffill())
        st.dataframe(forecast_df)

st.markdown("---")
st.caption("Smart Factory Dashboard • CNN Defect Detection + Predictive Maintenance + LSTM Forecasting • Streamlit")

    
    

    