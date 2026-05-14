import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIG ---
st.set_page_config(page_title="MHTCET College Predictor", page_icon="🎓", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stSelectbox, .stNumberInput {
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_edu = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_hzfmxos7.json") # Education/Search animation

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- MAPPINGS ---
# Note: Update these mappings if your training encoding was different
gender_map = {"Male": 0, "Female": 1}
category_map = {
    "OPEN": 0, "OBC": 1, "SC": 2, "ST": 3, 
    "VJ/DT": 4, "NT-1": 5, "NT-2": 6, "NT-3": 7, 
    "EWS": 8, "TFWS": 9
}

# --- HEADER ---
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🎓 MHTCET College Predictor")
        st.subheader("Find the best college based on your percentile and category.")
        st.write("Enter your details in the sidebar to get an instant prediction.")
    with col2:
        st_lottie(lottie_edu, height=200, key="edu")

st.divider()

# --- SIDEBAR INPUTS ---
st.sidebar.header("User Information")
percentile = st.sidebar.number_input("MHTCET Percentile", min_value=0.0, max_value=100.0, value=90.0, step=0.01)
gender = st.sidebar.selectbox("Gender", options=list(gender_map.keys()))
category = st.sidebar.selectbox("Category", options=list(category_map.keys()))

# --- PREDICTION LOGIC ---
if st.sidebar.button("Predict College"):
    # Prepare input data
    input_data = np.array([[
        percentile, 
        gender_map[gender], 
        category_map[category]
    ]])
    
    try:
        # Get prediction
        prediction = model.predict(input_data)
        
        # Display Results
        st.balloons()
        st.markdown("### 🎯 Predicted College:")
        st.markdown(f"""
            <div class="prediction-card">
                <h2 style='color: #ff4b4b;'>{prediction[0]}</h2>
                <p style='font-size: 1.2em; color: #555;'>Based on your profile, this college is the best fit according to our model.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Additional Info
        with st.expander("Show Technical Details"):
            st.write(f"**Input Percentile:** {percentile}")
            st.write(f"**Encoded Inputs:** {input_data}")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("Powered by Streamlit and Scikit-Learn | Built for Engineering Aspirants")
