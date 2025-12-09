import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from ultralytics import YOLO
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EcoGuard: Aerial AI",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ULTRA-MODERN CSS (THE UI UPGRADE) ---
st.markdown("""
    <style>
    /* 1. Animated 'Sunset Savannah' Background */
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .stApp {
        background: linear-gradient(-45deg, #FF512F, #DD2476, #1A2980, #26D0CE);
        background-size: 400% 400%;
        animation: gradient 12s ease infinite;
        font-family: 'Helvetica', sans-serif;
    }

    /* 2. Glassmorphism Containers (Fixes Text Visibility) */
    .glass-container {
        background: rgba(255, 255, 255, 0.90); /* High opacity white for readability */
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #1A2980; /* Dark blue text for contrast */
        margin-bottom: 20px;
    }

    /* 3. Metric Cards Styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 10px;
    }
    
    /* 4. Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.4);
    }
    
    /* 5. Header Styling */
    h1, h2, h3 {
        color: #ffffff;
        text-shadow: 2px 2px 4px #000000;
    }
    
    /* 6. Custom Button Styling */
    .stButton>button {
        background: linear-gradient(to right, #11998e, #38ef7d);
        color: white;
        border: none;
        border-radius: 30px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(56, 239, 125, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD MODELS ---
@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model('wildlife_v2_mobilenet.h5')

@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt') 

try:
    classifier = load_classifier()
    yolo_model = load_yolo()
except Exception as e:
    st.markdown('<div class="glass-container"><h3>⚠️ Model Error</h3><p>Please run the training notebook first to generate "wildlife_v2_mobilenet.h5".</p></div>', unsafe_allow_html=True)
    st.stop()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3069/3069172.png", width=100)
    st.title("⚙️ Control Panel")
    st.markdown("---")
    confidence_threshold = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.60)
    st.info("💡 **Tip:** Use the 'Population Count' tab to detect multiple animals in one frame.")

# --- 5. MAIN HEADER ---
st.title("🦅 EcoGuard: Aerial Analytics")
st.markdown("### 🌍 Autonomous Wildlife Census & Protection System")
st.markdown("---")

# --- 6. LAYOUT ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown('<div class="glass-container"><h3>📤 Input Feed</h3><p>Upload drone imagery below.</p></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop Image Here (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Source Feed', use_column_width=True, output_format="JPEG")

if uploaded_file:
    with col2:
        # We wrap the results in a glass container for perfect visibility
        st.markdown('<div class="glass-container"><h3>📊 Analysis Dashboard</h3>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🦁 Species Identification", "🔭 Population Count"])
        
        # --- TAB 1: CLASSIFICATION ---
        with tab1:
            with st.spinner('Processing Neural Network...'):
                # Resize and Predict
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = classifier.predict(img_array)
                classes = ['Buffalo', 'Elephant', 'Rhino', 'Zebra']
                predicted_class = classes[np.argmax(predictions)]
                confidence = np.max(predictions)
            
            # Result Card
            st.markdown(f"""
                <div class="metric-card">
                    <h2>DETECTED: {predicted_class.upper()}</h2>
                    <h3>Confidence: {confidence*100:.1f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Plotly Graph with Dark Theme
            fig = go.Figure(data=[go.Bar(
                x=classes,
                y=predictions[0],
                marker_color=['#FF6B6B', '#4ECDC4', '#FFE66D', '#1A535C']
            )])
            fig.update_layout(
                title='AI Confidence Distribution',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#1A2980'), # Dark Blue text for visibility on glass
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- TAB 2: COUNTING (YOLO) ---
        with tab2:
            st.info("Running YOLOv8 Object Detection...")
            
            # YOLO Logic
            yolo_results = yolo_model.predict(image, conf=0.35) 
            res_plotted = yolo_results[0].plot()
            res_image = Image.fromarray(res_plotted[..., ::-1])
            count = len(yolo_results[0].boxes)
            
            st.image(res_image, caption='Object Detection Output', use_column_width=True)
            
            st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                    <h2>Total Count: {count}</h2>
                    <p>Animals Detected in Frame</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True) # End Glass Container

else:
    # Empty State with Glass Effect
    with col2:
        st.markdown("""
        <div class="glass-container">
            <h3>Waiting for Data Stream...</h3>
            <p>Please upload an aerial image from the left panel to begin analysis.</p>
        </div>
        """, unsafe_allow_html=True)