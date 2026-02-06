import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
from PIL import Image
from groq import Groq
from supabase import create_client
import sqlite3
import hashlib
from datetime import datetime
import random
from dotenv import load_dotenv

# Load .env file
load_dotenv()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "module" not in st.session_state:
    st.session_state.module = "manufacturer"


# ---------------------------------------------------------
# 1. PAGE CONFIG & THEME ENGINE (The "React" Look)
# ---------------------------------------------------------

st.set_page_config(
    page_title="FactoryMind AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)



def inject_industrial_theme():
    # Aap yahan apni pasand ki image ka URL change kar sakte hain
    bg_img = "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"
    
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        /* Background Image with Dark Overlay */
        .stApp {{
            background: linear-gradient(rgba(14, 17, 23, 0.85), rgba(14, 17, 23, 0.85)), 
                        url("{bg_img}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: #E2E8F0; }}

        /* Glassmorphic Metric Cards */
        div[data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            padding: 20px !important;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }}

        /* Sidebar Modernization */
        section[data-testid="stSidebar"] {{
            background-color: rgba(11, 14, 20, 0.95) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .nav-item {{
            display: flex; align-items: center; padding: 12px 15px;
            margin: 8px 0; border-radius: 10px; color: #94A3B8;
            text-decoration: none; font-weight: 500; transition: 0.3s;
        }}
        .nav-item:hover {{ background: rgba(59, 130, 246, 0.2); color: #3B82F6; }}
        .nav-active {{ background: rgba(59, 130, 246, 0.3) !important; color: #FFFFFF !important; border-left: 4px solid #3B82F6; }}

        /* High-Tech Headers */
        .glow-text {{
            font-size: 3rem; font-weight: 800;
            background: linear-gradient(90deg, #60A5FA, #2DD4BF);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            text-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }}

        /* Styled Cards for Content */
        .content-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px; padding: 35px; margin-bottom: 40px auto;
            backdrop-filter: blur(15px);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            display: block;
        }}
        
        /* Buttons */
        .stButton>button {{
            width: 100%; border-radius: 12px; font-weight: 600;
            background: linear-gradient(90deg, #3B82F6 0%, #2563EB 100%);
            border: none; color: white; transition: 0.3s;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        }}
        .stButton>button:hover {{ transform: translateY(-2px); box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5); }}

        /* Input fields glass style */
        .stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
            background-color: rgba(255, 255, 255, 0.05) !important;
            border-radius: 10px !important;
            color: white !important;
        }}
        
    </style>
    """, unsafe_allow_html=True)

inject_industrial_theme()

# ---------------------------------------------------------
# 🔐 AUTH SYSTEM (SQLite Login)
# ---------------------------------------------------------
DB_FILE = "users.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT,
            password TEXT,
            role TEXT DEFAULT 'operator'
        )
    """)
    conn.commit()
    conn.close()


init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, email, password, role):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
            (username, email, hash_password(password), role)
        )
        conn.commit()
        conn.close()
        return True
    except:
        return False


def login_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "SELECT username, role FROM users WHERE username=? AND password=?",
        (username, hash_password(password))
    )
    user = c.fetchone()
    conn.close()
    return user


# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
def login_signup_page():
    # Heading ko center kiya
    st.markdown('<h1 class="glow-text" style="text-align: center;">FactoryMind AI Portal</h1>', unsafe_allow_html=True)
    
    # Columns for centering (CSS ke baghair best tareeka)
    left_spacer, center_column, right_spacer = st.columns([1, 2, 1])

    with center_column:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

        # ---------------- LOGIN TAB ----------------
        with tab1:
            # Keys ko unique rakhne ke liye suffix use kiya
            username = st.text_input("Username", key="auth_login_username")
            password = st.text_input("Password", type="password", key="auth_login_password")

            if st.button("Login", key="auth_login_btn"):
                user = login_user(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.role = user[1]
                    st.success("Login successful 🚀")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        # ---------------- SIGNUP TAB ----------------
        with tab2:
            # In keys ko change kar diya taake Duplicate Key error khatam ho jaye
            new_user = st.text_input("New Username", key="auth_signup_username_unique")
            role = st.selectbox("Role", ["operator", "maintenance", "manager"], key="auth_signup_role")
            new_email = st.text_input("Email", key="auth_signup_email_unique")
            new_pass = st.text_input("Password", type="password", key="auth_signup_password_unique")

            if st.button("Create Account", key="auth_signup_btn"):
                if create_user(new_user, new_email, new_pass, role):
                    st.success("Account created! Please login.")
                else:
                    st.error("Username already exists")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- SIGNUP TAB ----------------
    with tab2:
        new_user = st.text_input("New Username", key="signup_user")
        role = st.selectbox("Role", ["operator", "maintenance", "manager"])
        new_email = st.text_input("Email", key="signup_email")
        new_pass = st.text_input("Password", type="password", key="signup_pass")

        if st.button("Create Account", key="signup_btn"):
            if create_user(new_user, new_email, new_pass,role):
                st.success("Account created! Please login.")
            else:
                st.error("Username already exists")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. CORE UTILITIES & MODELS
# ---------------------------------------------------------
DEFECT_MODEL_FILE = "textile_model (2).h5"
PM_MODEL_FILE = "xgb_predictive_maintenance.joblib"
SCALER_FILES = ["scaler.joblib", "pm_scaler.joblib"]
LSTM_MODEL_FILE = "lstm_model.h5"
FORECAST_SCALER_FILE = "scaler_forecast.joblib"
FORECAST_CSV = "monthly_retail_sales_cleaned.csv"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DEFECT_CLASSES = ['Broken stitch', 'Needle mark', 'Pinched fabric', 'Vertical', 'defect free', 'hole', 'horizontal', 'lines', 'stain']
REQUIRED_FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if GROQ_API_KEY: client = Groq(api_key=GROQ_API_KEY)
if SUPABASE_URL and SUPABASE_KEY: supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else: supabase = None

# ------------------- SUPABASE / LIVE SENSOR HELPERS -------------------
def fetch_latest_machine_data():
    if not supabase:
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
        "Type": int(row["type"]),
        "Air temperature [K]": float(row["air_temp"]),
        "Process temperature [K]": float(row["process_temp"]),
        "Rotational speed [rpm]": float(row["rpm"]),
        "Torque [Nm]": float(row["torque"]),
        "Tool wear [min]": float(row["tool_wear"]),
        "Timestamp": row["created_at"]
    }

def insert_dummy_data():
    if not supabase:
        return None
    data = {
        "type": random.randint(0, 2),
        "air_temp": random.uniform(280, 360),
        "process_temp": random.uniform(290, 370),
        "rpm": random.uniform(1000, 1600),
        "torque": random.uniform(30, 50),
        "tool_wear": random.uniform(100, 220)
    }
    supabase.table("machine_telemetry").insert(data).execute()
    return data

def get_machine_status(machine_data):
    status = "Healthy"
    if machine_data["Tool wear [min]"] > 200 or machine_data["Process temperature [K]"] > 350:
        status = "Critical"
    elif machine_data["Tool wear [min]"] > 150 or machine_data["Process temperature [K]"] > 320:
        status = "Warning"
    return status


# Caching Models
@st.cache_resource
def load_models():
    d_model = tf.keras.models.load_model(DEFECT_MODEL_FILE, compile=False) if os.path.exists(DEFECT_MODEL_FILE) else None
    pm_model = joblib.load(PM_MODEL_FILE) if os.path.exists(PM_MODEL_FILE) else None
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_FILE, compile=False) if os.path.exists(LSTM_MODEL_FILE) else None
    return d_model, pm_model, lstm_model

d_model, pm_model, lstm_model = load_models()

# Helper Functions
def log_to_csv(file_path, data_dict):
    df = pd.DataFrame([data_dict])
    full_path = os.path.join(DATA_DIR, file_path)
    if os.path.exists(full_path):
        pd.concat([pd.read_csv(full_path), df], ignore_index=True).to_csv(full_path, index=False)
    else:
        df.to_csv(full_path, index=False)

def encode_image(image_pil):
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def top_right_controls():
    col_spacer, col1, col2, col3 = st.columns([6, 1, 1, 1])

    with col1:
        if st.button("Home", key="nav_home"):
            st.session_state.module = "manufacturer"
            st.rerun()

    with col2:
        if st.button("Contact", key="nav_contact"):
            st.session_state.module = "pm"
            st.rerun()

    with col3:
        if st.button("Logout", key="nav_logout"):
            st.session_state.logged_in = False
            st.session_state.module = "manufacturer"
            st.rerun()

# ---------------------------------------------------------
# 🔒 AUTH CHECK BEFORE DASHBOARD LOADS
# ---------------------------------------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login_signup_page()
    st.stop()
    
st.markdown("<div style='margin-top:-60px'></div>", unsafe_allow_html=True)
top_right_controls()


# ---------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ---------------------------------------------------------
MENU = [
    ("manufacturer", "🏭 Dashboard"),
    ("defect", "🩻 Vision AI"),
    ("pm", "⚙️ Maintenance"),
    ("forecast", "📈 Forecasting")
]

active_key = st.session_state.module


with st.sidebar:
    st.markdown(
        '<h2 style="color:white; margin-bottom:20px; font-family: "Inter", sans-serif;">FactoryMind'
        '<span style="color:#3B82F6;">AI</span></h2>',
        unsafe_allow_html=True
    )

    # Custom CSS for the Advanced Smart Factory look
    st.markdown("""
    <style>
        /* Container for the nav items */
        .nav-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        /* The Nav Item Styling */
        .nav-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 12px 20px;
            color: #94A3B8;
            text-decoration: none;
            display: flex;
            align-items: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        /* Hover effect: Glow and Slide */
        .nav-card:hover {
            background: rgba(59, 130, 246, 0.1);
            border-color: #3B82F6;
            color: #FFFFFF;
            transform: translateX(5px);
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
        }

        /* Active State: Highlight with gradient border */
        .nav-active {
            background: linear-gradient(90deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0) 100%);
            border-left: 4px solid #3B82F6 !important;
            color: #FFFFFF !important;
        }

        .nav-icon {
            margin-right: 15px;
            font-size: 1.2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Render Navigation
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    for key, label in MENU:
        # Determine if this item is active
        is_active = "nav-active" if st.session_state.module == key else ""
        
        # We use a button hidden inside a div or a link to trigger st.rerun
        if st.button(label, key=f"nav_{key}", use_container_width=True, type="secondary"):
            st.session_state.module = key
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("v2.2.0-Stable | Built for Scale")



# ---------------------------------------------------------
# 4. PAGE ROUTING
# ---------------------------------------------------------

# --- HOME DASHBOARD ---
if active_key == "manufacturer":
    st.markdown('<h1 class="glow-text">Factory Command Center</h1>', unsafe_allow_html=True)
    
    # Live KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("OEE Status", "84.2%", "+1.2%")
    col2.metric("Active Lines", "26/32", "Stable")
    col3.metric("Quality Rate", "99.1%", "+0.05%")
    col4.metric("Energy Use", "1.2MW", "-10%")
    col5.metric("Cycle Time", "42s", "-2s")

    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    c_left, c_right = st.columns([2, 1])
    with c_left:
        st.subheader("📊 Real-time Production Volume")
        chart_data = pd.DataFrame(np.random.randn(20, 2), columns=['Line A', 'Line B'])
        st.line_chart(chart_data)
        with c_right:
            st.subheader("🔔 System Alerts")
            st.warning("Line 4: Tool wear exceeding 85%")
            st.error("Line 12: Motor temperature critical")
            st.success("Line 1: Batch completed successfully")
        st.markdown('</div>', unsafe_allow_html=True)

# --- VISION AI LAB ---
elif active_key == "defect":
    st.markdown('<h1 class="glow-text">Textile Defect Detection</h1>', unsafe_allow_html=True)
    
    col_up, col_res = st.columns([1, 1.2])
    
    with col_up:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload High-Res Fabric Sample", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, use_container_width=True, caption="Input Scan")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        if uploaded_file and d_model:
            img_resized = img.resize((224, 224))
            arr = np.array(img_resized).astype("float32") / 255.0
            preds = d_model.predict(np.expand_dims(arr, axis=0))
            class_idx = np.argmax(preds)
            class_name = DEFECT_CLASSES[class_idx]
            conf = np.max(preds)

            st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.15); border-left: 5px solid #3B82F6; padding: 25px; border-radius: 15px; backdrop-filter: blur(10px);">
                <h3 style="margin:0; color: white;">Result: {class_name}</h3>
                <p style="margin:5px 0 0 0; opacity: 0.8; font-size: 1.1rem;">AI Confidence: {conf*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            if GROQ_API_KEY:
                with st.spinner("AI Expert analyzing defect..."):
                    prompt = f"""
                    You are a textile manufacturing expert AI.

                    Defect detected: {class_name}
                    Confidence: {conf*100:.2f}%

                    Give:
                    1. Root cause
                    2. Severity level
                    3. Fix recommendation
                    4. Prevention tips

                    Answer in short industrial format.
                    """

                    try:
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.4
                        )

                        ai_text = response.choices[0].message.content

                        st.markdown('<div class="content-card" style="margin-top:20px;">', unsafe_allow_html=True)
                        st.markdown("### 🧠 AI Expert Analysis")
                        st.write(ai_text)
                        st.markdown('</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Groq error: {e}")

# --- PREDICTIVE MAINTENANCE ---
elif active_key == "pm":
    pm_scaler = joblib.load("pm_scaler.joblib")
    pm_model = joblib.load("xgb_predictive_maintenance.joblib")
    st.markdown('<h1 class="glow-text">Predictive Maintenance</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("🔧 Machine Digital Twin Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        air = st.number_input("Air Temp [K]", value=300.0)
        proc = st.number_input("Process Temp [K]", value=310.0)
    with c2:
        rpm = st.number_input("Rotational Speed [RPM]", value=1500.0)
        torque = st.number_input("Torque [Nm]", value=40.0)
    with c3:
        wear = st.number_input("Tool Wear [min]", value=50.0)
        m_type = st.selectbox("Machine Type", [0, 1, 2])

    if st.button("Calculate Failure Probability"):
        feature_dict = {
            "Type": m_type,
            "Air temperature [K]": air,
            "Process temperature [K]": proc,
            "Vibration": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
        }
        features = np.array([[feature_dict[f] for f in REQUIRED_FEATURES]])
        st.write("🔍 Model Input Vector:", features)

        features_scaled = pm_scaler.transform(features)
        prob = pm_model.predict_proba(features_scaled)[0][1]
        if prob >= 0.6:
            st.error(f"🔴 CRITICAL: {prob*100:.1f}% failure risk. Stop machine immediately.")
        elif prob >= 0.3:
            st.warning(f"🟡 WARNING: {prob*100:.1f}% failure risk. Schedule maintenance.")
        else:
            st.success(f"🟢 HEALTHY: {prob*100:.1f}% failure risk.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("📡 Live Machine Telemetry")

    # Button to insert dummy data for testing
    if st.button("Generate Dummy Data for Testing"):
        dummy = insert_dummy_data()
        if dummy:
            st.success("✅ Dummy data inserted")
            st.write(dummy)
        else:
            st.error("Supabase not configured!")

    # Fetch latest telemetry
    machine_data = fetch_latest_machine_data()
    if machine_data:
        st.dataframe(pd.DataFrame([machine_data]).drop(columns=["Timestamp"]))
        status = get_machine_status(machine_data)
        if status == "Healthy":
            st.success(f"✅ Status: {status}")
        elif status == "Warning":
            st.warning(f"⚠️ Status: {status}")
        else:
            st.error(f"❌ Status: {status}")
    else:
        st.info("ℹ️ No telemetry found. Generate dummy data or connect live sensors.")

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------
# LSTM Forecasting
# -----------------------

elif active_key == "forecast":
    st.markdown('<h1 class="glow-text">Demand & Production Forecast</h1>', unsafe_allow_html=True)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    # File paths check
    if not os.path.exists(LSTM_MODEL_FILE) or not os.path.exists(FORECAST_SCALER_FILE):
        st.error("Missing model (.h5) or scaler (.joblib) files.")
        st.stop()

    # --- DIRECT LOADING (Jaise aapne manga) ---
    # LSTM model ke liye tf.keras aur Scaler ke liye joblib
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_FILE, compile=False)
    forecast_scaler = joblib.load(FORECAST_SCALER_FILE)

    # Data Loading
    if os.path.exists(FORECAST_CSV):
        df_hist = pd.read_csv(FORECAST_CSV)
        df_hist = df_hist.rename(columns={"month": "Date", "Monthly_Sales": "Production"})
        df_hist["Date"] = pd.to_datetime(df_hist["Date"], errors="coerce")
        df_hist = df_hist.sort_values("Date").reset_index(drop=True)
        series = df_hist.set_index("Date")["Production"].astype(float)

        # UI Inputs
        col1, col2 = st.columns(2)
        with col1:
            seq_len = st.number_input("Sequence length (Lookback)", min_value=1, max_value=60, value=12)
        with col2:
            months = st.slider("Months ahead", 1, 24, 12)

        if len(series) >= seq_len:
            # Prediction Logic
            last_vals = series.values[-seq_len:].reshape(-1, 1)
            scaled_seq = forecast_scaler.transform(last_vals).reshape(1, seq_len, 1)

            preds_scaled = []
            seq = scaled_seq.copy()
            for _ in range(months):
                pred_scaled = lstm_model.predict(seq, verbose=0)[0][0]
                preds_scaled.append(pred_scaled)
                # Sequence update: Purana data hatao, naya prediction add karo
                seq = np.append(seq[:, 1:, :], [[[pred_scaled]]], axis=1)

            # Rescale Predictions
            preds_rescaled = forecast_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
            
            # Date Range Generation
            last_date = series.index[-1]
            future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=months, freq="MS")
            forecast_df = pd.DataFrame({"Forecast": preds_rescaled}, index=future_dates)

            # Visuals
            st.subheader("📊 Historical vs. Forecasted Production")
            combined = pd.concat([series, forecast_df["Forecast"]], axis=1)
            combined.columns = ["Historical", "Forecast"]
            st.line_chart(combined.ffill())
            
            # Dynamic Info Box
            pct_change = ((preds_rescaled.mean() - series.mean()) / series.mean()) * 100
            st.info(f"The LSTM model predicts a {pct_change:.1f}% change in average production for the next {months} months.")
            
            with st.expander("📄 View Forecast Table"):
                st.dataframe(forecast_df)
        else:
            st.warning(f"Not enough data. Please provide at least {seq_len} months of history.")
    else:
        st.error(f"CSV file '{FORECAST_CSV}' not found.")

    st.markdown('</div>', unsafe_allow_html=True)