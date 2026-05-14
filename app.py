import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from streamlit_lottie import st_lottie

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MHTCET College Predictor", 
    page_icon="🎓", 
    layout="wide"
)

# --- 2. CUSTOM CSS FOR ATTRACTIVE LAYOUT ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border: none;
    }
    .prediction-card {
        padding: 30px;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 10px solid #007bff;
    }
    .result-header {
        color: #1c1c1c;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SAFE ANIMATION LOADER (RESOLVES YOUR ERROR) ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        # Returns None if there's a network error or invalid JSON
        return None

# Reliable Lottie animation link
lottie_url = "https://lottie.host/82544e3d-080c-4976-96b6-397a6e11894a/9M6XnZOfn3.json"
lottie_data = load_lottieurl(lottie_url)

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'model.pkl' not found. Please ensure it's in the same directory.")
        return None

model = load_model()

# --- 5. MAPPINGS (Adjust based on your training labels) ---
gender_map = {"Male": 0, "Female": 1}
category_map = {
    "OPEN": 0, "OBC": 1, "SC": 2, "ST": 3, 
    "VJ/DT": 4, "NT-1": 5, "NT-2": 6, "NT-3": 7, 
    "EWS": 8, "TFWS": 9
}

# --- 6. HEADER SECTION ---
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🎓 Engineering College Predictor")
        st.write("""
            Welcome! This tool uses Machine Learning (KNN) to suggest the most likely 
            college allotment based on your MHTCET percentile and category.
        """)
    with col2:
        # Safety check for Lottie: Only render if data exists
        if lottie_data:
            st_lottie(lottie_data, height=180, key="coding")
        else:
            st.title("🏛️")

st.divider()

# --- 7. SIDEBAR INPUTS ---
st.sidebar.header("Enter Your Scores")
percentile = st.sidebar.number_input("MHTCET Percentile", min_value=0.0, max_value=100.0, value=90.0)
gender = st.sidebar.selectbox("Gender", options=list(gender_map.keys()))
category = st.sidebar.selectbox("Category", options=list(category_map.keys()))

predict_btn = st.sidebar.button("Predict Best College")

# --- 8. MAIN CONTENT / RESULTS ---
if predict_btn:
    if model:
        # Encode inputs
        input_data = np.array([[
            percentile, 
            gender_map[gender], 
            category_map[category]
        ]])
        
        try:
            # Prediction
            prediction = model.predict(input_data)
            
            st.balloons()
            
            # Layout for results
            st.markdown("### 🎯 Prediction Results")
            
            # Metrics for a dashboard feel
            m1, m2, m3 = st.columns(3)
            m1.metric("Percentile", f"{percentile}%")
            m2.metric("Category", category)
            m3.metric("Gender", gender)
            
            st.markdown(f"""
                <div class="prediction-card">
                    <p style='color: #666; text-transform: uppercase; letter-spacing: 1px;'>Recommended Institute</p>
                    <h2 class="result-header">{prediction[0]}</h2>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Model not loaded correctly.")

else:
    # Placeholder when no prediction has been made yet
    st.info("Adjust the values in the sidebar and click **Predict** to see your results.")

# --- 9. FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("© 2026 College Prediction System | Designed for Career Guidance")
