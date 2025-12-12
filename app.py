import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from ultralytics import YOLO
import time

st.set_page_config(
    page_title="EcoGuard Enterprise",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* 1. BACKGROUND: Clean Slate-Blue Gradient (Professional) */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    }

    /* 2. CONTAINERS: Pure White with Strong Shadows */
    .glass-container {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-top: 5px solid #2980b9; /* Professional Blue Top Border */
        margin-bottom: 20px;
    }

    /* 3. TEXT VISIBILITY UTILITIES */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important; /* Dark Navy Blue for Headers */
        font-weight: 700;
    }
    p, div, label, span, li {
        color: #000000 !important; /* Pure Black for body text */
        font-weight: 500;
    }
    
    /* 4. METRIC CARDS (Clean White & Blue) */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #dcdcdc;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* 5. POACHER ALERT (High Contrast Red) */
    .poacher-alert {
        background-color: #e74c3c; /* Bright Red */
        color: #ffffff !important; /* White Text */
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 22px;
        box-shadow: 0 4px 10px rgba(231, 76, 60, 0.4);
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    /* 6. SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #bdc3c7;
    }
    
    /* 7. TITLES (Exceptions for the main gradient background) */
    .main-title {
        color: #2c3e50 !important;
        text-shadow: 1px 1px 0px rgba(255,255,255,0.5);
    }
    .sub-title {
        color: #34495e !important;
    }
    
    /* 8. BUTTONS */
    .stButton>button {
        background-color: #2980b9;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3498db;
    }
    </style>
    """, unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_models():
    try:
        classifier = tf.keras.models.load_model('wildlife_v2_mobilenet.h5')
        yolo = YOLO('yolov8n.pt') 
        return classifier, yolo
    except:
        return None, None

classifier, yolo_model = load_models()


def process_image(image, conf_thresh, iou_thresh):
    """Detects People & Animals -> Crops -> Classifies"""
    results = yolo_model.predict(image, conf=conf_thresh, iou=iou_thresh, verbose=False)
    boxes = results[0].boxes
    
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    
    counts = {'Person': 0, 'Elephant': 0, 'Rhino': 0, 'Buffalo': 0, 'Zebra': 0}
    poacher_detected = False
    
    animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23] 
    person_class = 0 
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0])
        
        if cls == person_class:
            counts['Person'] += 1
            poacher_detected = True
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            draw.text((x1, y1-20), " HUMAN DETECTED ", fill="white", stroke_fill="red", stroke_width=2)
            
        elif cls in animal_classes:
            crop = image.crop((x1, y1, x2, y2))
            
            if crop.size[0] > 0 and crop.size[1] > 0:
                img_resized = crop.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = classifier.predict(img_array, verbose=0)
                classes = ['Buffalo', 'Elephant', 'Rhino', 'Zebra']
                predicted_species = classes[np.argmax(predictions)]
                confidence = np.max(predictions)
                
                counts[predicted_species] += 1
                
                draw.rectangle([x1, y1, x2, y2], outline="#00aa00", width=3)
                label = f" {predicted_species} "
                draw.text((x1, y1-20), label, fill="white", stroke_fill="#00aa00", stroke_width=2)

    return draw_img, counts, poacher_detected

def generate_report(counts, risk):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
    ECOGUARD ENTERPRISE - DAILY LOG
    Date: {timestamp}
    -------------------------------
    THREAT STATUS: {risk}
    
    CENSUS COUNT:
    - People Detected: {counts['Person']}
    - Elephants: {counts['Elephant']}
    - Rhinos: {counts['Rhino']}
    - Buffalos: {counts['Buffalo']}
    - Zebras: {counts['Zebra']}
    
    Auto-Generated Report.
    """
    return report

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3069/3069172.png", width=80)
    st.markdown("### ⚙️ System Settings")
    st.markdown("---")
    
    st.markdown("**🎯 Sensitivity Control**")
    conf_thresh = st.slider("Detection Confidence", 0.1, 0.9, 0.30)
    iou_thresh = st.slider("Overlap Threshold", 0.1, 0.9, 0.45)
    
    st.markdown("---")
    st.info("Status: System Online")

st.markdown('<h1 class="main-title">🛡️ EcoGuard Enterprise</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-title">Multi-Species Census & Anti-Poaching System</h3>', unsafe_allow_html=True)
st.markdown("---")

if classifier is None:
    st.error("⚠️ System Offline: Model files missing.")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="glass-container"><h3>📡 Drone Uplink</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Feed", type=['jpg','png','jpeg'], label_visibility='collapsed')
    
    if uploaded_file:
        raw_image = Image.open(uploaded_file).convert("RGB")
        st.image(raw_image, caption="Source Feed", use_column_width=True)
        st.caption(f"Resolution: {raw_image.size[0]}x{raw_image.size[1]}px")
    else:
        st.info("Waiting for video/image feed...")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    with col2:
        st.markdown('<div class="glass-container"><h3>🧠 Tactical Analysis</h3>', unsafe_allow_html=True)
        
        with st.spinner('Processing sector...'):
            final_img, counts, poacher_alert = process_image(raw_image, conf_thresh, iou_thresh)
            
            if poacher_alert:
                st.markdown(f'<div class="poacher-alert">🚨 SECURITY ALERT: {counts["Person"]} PERSON DETECTED 🚨</div>', unsafe_allow_html=True)
                risk_level = "CRITICAL (HUMAN INCURSION)"
            else:
                risk_level = "SAFE"
                st.markdown('<div style="background-color:#d4edda; padding:15px; border-radius:10px; color:#155724; text-align:center; margin-bottom:15px;">✅ Sector Clear: No unauthorized humans.</div>', unsafe_allow_html=True)

            st.image(final_img, caption="AI Identification Grid", use_column_width=True)
            
            st.markdown("#### 📊 Census Metrics")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Rhinos", counts['Rhino'])
            c2.metric("Elephants", counts['Elephant'])
            c3.metric("Zebras", counts['Zebra'])
            
            c4, c5, c6 = st.columns(3)
            c4.metric("Buffalos", counts['Buffalo'])
            c5.metric("People", counts['Person'], delta="-ALERT" if counts['Person']>0 else "None", delta_color="inverse")
            c6.metric("Total Count", sum(counts.values()))

            chart_data = pd.DataFrame({
                'Species': list(counts.keys()),
                'Count': list(counts.values())
            })
            
            fig = go.Figure([go.Bar(
                x=chart_data['Species'], 
                y=chart_data['Count'],
                marker_color=['#e74c3c', '#3498db', '#3498db', '#3498db', '#3498db'], # Red for Person, Blue for others
                text=chart_data['Count'],
                textposition='auto'
            )])
            fig.update_layout(
                title="Population Distribution",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                font=dict(color='black')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            report = generate_report(counts, risk_level)
            st.download_button("📥 Download Sector Report", report, file_name="sector_log.txt")

        st.markdown('</div>', unsafe_allow_html=True)