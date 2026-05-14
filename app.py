import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="College Predictor Pro", 
    page_icon="📊", 
    layout="wide"
)

# --- 2. CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .prediction-output {
        padding: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ASSET LOADERS ---
def load_lottieurl(url):
    try:
        return requests.get(url, timeout=5).json()
    except:
        return None

lottie_chart = load_lottieurl("https://lottie.host/82544e3d-080c-4976-96b6-397a6e11894a/9M6XnZOfn3.json")

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# --- 4. DATA MAPPINGS ---
gender_map = {"Male": 0, "Female": 1}
category_map = {
    "OPEN": 0, "OBC": 1, "SC": 2, "ST": 3, 
    "VJ/DT": 4, "NT-1": 5, "NT-2": 6, "NT-3": 7, 
    "EWS": 8, "TFWS": 9
}

# --- 5. SIDEBAR (VERTICAL OPTIONS) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3429/3429149.png", width=100)
st.sidebar.title("Student Profile")
st.sidebar.write("Fill in your details below:")

# Vertical Inputs
percentile = st.sidebar.slider("MHTCET Percentile", 0.0, 100.0, 85.0, 0.01)
gender = st.sidebar.radio("Select Gender", options=list(gender_map.keys()))
category = st.sidebar.selectbox("Select Category", options=list(category_map.keys()))

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Generate Prediction")

# --- 6. MAIN CONTENT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🚀 College Admission Insights")
    st.info("This system uses a KNN Algorithm to match your score with historical admission data.")

with col2:
    if lottie_chart:
        st_lottie(lottie_chart, height=150)

if predict_btn:
    # Model Prediction
    input_features = np.array([[percentile, gender_map[gender], category_map[category]]])
    prediction = model.predict(input_features)
    
    # --- VISUALIZATION SECTION ---
    st.subheader("📊 Admission Analytics")
    
    tab1, tab2 = st.tabs(["Prediction Result", "Score Analysis"])
    
    with tab1:
        st.markdown(f"""
            <div class="prediction-output">
                <p style='font-size: 1.2rem; opacity: 0.9;'>Predicted Allotment:</p>
                <h1 style='font-size: 2.5rem;'>{prediction[0]}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Percentile", f"{percentile}%")
        c2.metric("Category", category)
        c3.metric("Gender", gender)

    with tab2:
        # Gauge Chart for Percentile
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = percentile,
            title = {'text': "Percentile Standing"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#764ba2"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcfcf"},
                    {'range': [50, 85], 'color': "#f1f1f1"},
                    {'range': [85, 100], 'color': "#e1ffcf"}
                ]
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

else:
    # Welcome Visual
    st.write("### How it works")
    st.write("1. **Input:** Provide your percentile and category.")
    st.write("2. **Analyze:** The model compares your score against 300+ colleges.")
    st.write("3. **Result:** Get the most likely college name instantly.")
    
    # Generic bar chart for visual appeal
    example_data = pd.DataFrame({
        'Category': ['OPEN', 'OBC', 'SC', 'ST', 'EWS'],
        'Avg Percentile': [92, 88, 75, 65, 89]
    })
    fig_ex = px.bar(example_data, x='Category', y='Avg Percentile', title="Average Cutoff Trends (Example Data)")
    st.plotly_chart(fig_ex, use_container_width=True)

st.divider()
st.caption("Developed for Academic Guidance | Data-Driven Career Planning")
