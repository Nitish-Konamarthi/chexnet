import io
import os
import collections
from typing import List, Tuple
import sys

import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import pandas as pd

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import the DenseNet class from the cloned repo; adjust path if needed
try:
    from DensenetModels import DenseNet121
    from HeatmapGenerator import generate_gradcam, HeatmapGenerator
except Exception as e:
    st.error(f"Could not import modules: {str(e)}")
    raise

st.set_page_config(page_title="Chest X-ray Analysis Prototype", layout="wide")
#changed model path here
MODEL_PATH_ENV = os.environ.get("MODEL_PATH", "chexnet/models/m-30012020-104001.pth.tar")

#------------------------------------------------------------------------------------
# Clinical Configuration for Better Confidence Reporting
#------------------------------------------------------------------------------------

CLASS_NAMES = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule',
               'Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema',
               'Fibrosis','Pleural_Thickening','Hernia']

# Optimized thresholds based on CheXNet paper (per-disease)
CLINICAL_THRESHOLDS = {
    'Atelectasis': 0.45,
    'Cardiomegaly': 0.55,
    'Effusion': 0.50,
    'Infiltration': 0.35,
    'Mass': 0.60,
    'Nodule': 0.55,
    'Pneumonia': 0.40,
    'Pneumothorax': 0.50,
    'Consolidation': 0.45,
    'Edema': 0.55,
    'Emphysema': 0.60,
    'Fibrosis': 0.55,
    'Pleural_Thickening': 0.45,
    'Hernia': 0.70
}

#------------------------------------------------------------------------------------
# Helper Functions for Clinical Interpretation
#------------------------------------------------------------------------------------

def classify_confidence(probability, thresholds={'high': 0.65, 'medium': 0.40, 'low': 0.20}):
    """Classify prediction confidence level"""
    if probability >= thresholds['high']:
        return 'HIGH', '🔴'
    elif probability >= thresholds['medium']:
        return 'MEDIUM', '🟡'
    elif probability >= thresholds['low']:
        return 'LOW', '🟢'
    else:
        return 'MINIMAL', '⚪'


def get_clinical_summary(results, high_threshold=0.65, medium_threshold=0.40, low_threshold=0.20):
    """Generate clinical interpretation summary"""
    high_findings = [(name, prob) for name, prob in results if prob >= high_threshold]
    medium_findings = [(name, prob) for name, prob in results if medium_threshold <= prob < high_threshold]
    low_findings = [(name, prob) for name, prob in results if low_threshold <= prob < medium_threshold]
    
    if len(high_findings) == 0 and len(medium_findings) == 0:
        return "✅ **NORMAL** - No significant pathology detected above clinical threshold.", "NORMAL", high_findings, medium_findings, low_findings
    elif len(high_findings) > 0:
        diseases = ", ".join([name for name, _ in high_findings])
        return f"⚠️ **ABNORMAL - HIGH CONFIDENCE** - Detected: {diseases}. Recommend clinical correlation.", "ABNORMAL", high_findings, medium_findings, low_findings
    else:
        diseases = ", ".join([name for name, _ in medium_findings])
        return f"⚡ **ABNORMAL - MODERATE CONFIDENCE** - Possible findings: {diseases}. Consider further evaluation.", "BORDERLINE", high_findings, medium_findings, low_findings


def get_binary_prediction(probability, disease_name, use_optimized=True):
    """Determine if finding is positive using optimized thresholds"""
    if use_optimized and disease_name in CLINICAL_THRESHOLDS:
        threshold = CLINICAL_THRESHOLDS[disease_name]
    else:
        threshold = 0.5
    
    return probability >= threshold, threshold

#------------------------------------------------------------------------------------
# UI Header
#------------------------------------------------------------------------------------

st.title("🩺 Chest X-ray Analysis with Clinical Interpretation")

st.markdown("""
**Enhanced DenseNet121** chest X-ray pathology detection with:
- 🎯 Clinical confidence levels (High/Medium/Low)
- 🔥 Grad-CAM++ attention visualization
- 📊 Optimized disease-specific thresholds
""")

#------------------------------------------------------------------------------------
# Sidebar Configuration
#------------------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings & Configuration")
    
    model_path = st.text_input("Model path", value=MODEL_PATH_ENV)
    
    st.divider()
    st.subheader("📊 Confidence Thresholds")
    st.markdown("Adjust what counts as High/Medium/Low confidence:")
    high_conf_threshold = st.slider("High Confidence", 0.50, 0.95, 0.70, 0.05, 
                                     help="Findings above this are highly likely present")
    medium_conf_threshold = st.slider("Medium Confidence", 0.20, 0.70, 0.40, 0.05,
                                       help="Findings above this are possibly present")
    low_conf_threshold = st.slider("Low Confidence", 0.05, 0.40, 0.20, 0.05,
                                    help="Findings above this are worth noting")
    
    st.divider()
    st.subheader("🔬 Visualization Options")
    show_heatmap = st.checkbox("Show Grad-CAM++ Attention Map", value=True)
    heatmap_blend_alpha = st.slider("Heatmap intensity", 0.0, 1.0, 0.5, 0.05)
    
    st.divider()
    st.subheader("🎚️ Display Settings")
    show_all_diseases = st.checkbox("Show all 14 diseases", value=False)
    use_optimized_thresholds = st.checkbox("Use disease-specific thresholds", value=True, 
                                            help="Each disease has an optimized threshold based on research")
    top_k = st.number_input("Max findings to highlight", min_value=1, max_value=14, value=5)
    
    st.divider()
    st.markdown("### ℹ️ Confidence Levels")
    st.markdown("""
    - 🔴 **HIGH** (≥65%): Likely present
    - 🟡 **MEDIUM** (40-65%): Possibly present
    - 🟢 **LOW** (20-40%): Unlikely but notable
    - ⚪ **MINIMAL** (<20%): Not detected
    """)

#------------------------------------------------------------------------------------
# Model Loading (Cached)
#------------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)

def load_model(path: str):
    """Load DenseNet121 model from checkpoint with proper device handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Place the checkpoint or set MODEL_PATH.")
    
    # Load checkpoint with CPU mapping for portability
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    
    # Remove DataParallel wrapper prefix if present
    new_state = collections.OrderedDict()
    for k, v in state.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v

    # Initialize model and load state dict
    model = DenseNet121(14, True)
    model.load_state_dict(new_state, strict=False)
    
    # Setup device and move model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, os.path.basename(path), device


@st.cache_resource(show_spinner=False)
def load_heatmap_generator(path: str):
    """Load HeatmapGenerator for Grad-CAM++ visualization."""
    try:
        h = HeatmapGenerator(path, "DENSE-NET-121", 14, 224)
        return h
    except Exception as e:
        st.warning(f"Could not load HeatmapGenerator: {e}. CAM visualization will be disabled.")
        return None


# Load model (cached)
try:
    model, model_version, device = load_model(model_path)
    st.sidebar.success(f"✅ Model loaded: {model_version}\n💻 Device: {device}")
except Exception as e:
    st.sidebar.error(f"❌ Model load failed: {str(e)}")
    st.stop()

# Optionally load heatmap generator
heatmap_generator = None
if show_heatmap:
    heatmap_generator = load_heatmap_generator(model_path)

#------------------------------------------------------------------------------------
# Image Upload & Display
#------------------------------------------------------------------------------------

st.subheader("📤 Upload Chest X-ray Image")
uploaded = st.file_uploader("Select a PNG/JPG/TIFF chest X-ray image", type=["png","jpg","jpeg","tiff"])

if uploaded is None:
    st.info("👆 Upload an X-ray image to begin analysis.")
    st.stop()

# Read and prepare image
image_bytes = uploaded.read()
image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
image_rgb = Image.open(io.BytesIO(image_bytes)).convert("RGB")

# Display uploaded image
col1, col2 = st.columns(2)
with col1:
    st.image(image, caption="📸 Uploaded X-ray", use_column_width=True)

#------------------------------------------------------------------------------------
# Preprocessing & Prediction
#------------------------------------------------------------------------------------

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.485,0.485],[0.229,0.229,0.229])
])


def predict_pil(img: Image.Image, model, device) -> Tuple[List[Tuple[str, float]], torch.Tensor]:
    """Preprocess image and run inference using the model."""
    # Convert grayscale to RGB by stacking channels
    arr = np.stack([np.array(img)]*3, axis=-1).astype(np.uint8)
    inp = preprocess(Image.fromarray(arr)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(inp)
        probs = torch.sigmoid(out).squeeze(0).cpu().numpy()
    
    # Return sorted predictions
    pairs = sorted(zip(CLASS_NAMES, probs.tolist()), key=lambda x: x[1], reverse=True)
    return pairs, inp


def generate_gradcam_visualization(image_rgb: Image.Image, model, inp, device, blend_alpha=0.5) -> np.ndarray:
    """Wrapper for generate_gradcam from HeatmapGenerator module with dynamic blend control."""
    return generate_gradcam(model, inp, device, image_rgb, blend_alpha)

#------------------------------------------------------------------------------------
# Run Analysis Button
#------------------------------------------------------------------------------------

if st.button("🚀 Run Pathology Analysis", key="run_pred", use_container_width=True, type="primary"):
    try:
        with st.spinner("🔬 Analyzing X-ray..."):
            # Get predictions
            results, inp = predict_pil(image, model, device)
        
        # Generate clinical summary
        thresholds = {'high': high_conf_threshold, 'medium': medium_conf_threshold, 'low': low_conf_threshold}
        summary_text, assessment_level, high_findings, medium_findings, low_findings = get_clinical_summary(
            results, high_conf_threshold, medium_conf_threshold, low_conf_threshold
        )
        
        # Generate Grad-CAM++ if enabled
        cam_img = None
        if show_heatmap and heatmap_generator is not None:
            try:
                with st.spinner("🔥 Generating Grad-CAM++ attention map..."):
                    cam_img = generate_gradcam_visualization(image_rgb, model, inp, device, heatmap_blend_alpha)
            except Exception as e:
                st.warning(f"Could not generate Grad-CAM++: {e}")
        
        # Display Grad-CAM++ heatmap in column 2 if available
        if cam_img is not None:
            with col2:
                st.image(cam_img, caption="🔥 Grad-CAM++: AI Attention Map", use_column_width=True)
        
        st.divider()
        
        #------------------------------------------------------------------------------------
        # Clinical Summary Section
        #------------------------------------------------------------------------------------
        
        st.subheader("📋 Clinical Assessment")
        
        # Main summary box
        if assessment_level == "NORMAL":
            st.success(summary_text)
        elif assessment_level == "ABNORMAL":
            st.error(summary_text)
        else:
            st.warning(summary_text)
        
        st.divider()
        
        #------------------------------------------------------------------------------------
        # High Confidence Findings
        #------------------------------------------------------------------------------------
        
        if high_findings:
            st.markdown("### 🔴 High Confidence Findings")
            st.markdown(f"*Probability ≥ {high_conf_threshold*100:.0f}% - Highly likely present*")
            
            for disease, prob in high_findings[:top_k]:
                col_a, col_b, col_c = st.columns([4, 2, 2])
                with col_a:
                    # Check if positive using optimized threshold
                    is_positive, opt_threshold = get_binary_prediction(prob, disease, use_optimized_thresholds)
                    status = "✅ POSITIVE" if is_positive else "⚠️ BORDERLINE"
                    st.markdown(f"**{disease}** {status}")
                    if use_optimized_thresholds:
                        st.caption(f"Clinical threshold: {opt_threshold*100:.0f}%")
                with col_b:
                    st.metric("Probability", f"{prob*100:.1f}%")
                with col_c:
                    confidence, icon = classify_confidence(prob, thresholds)
                    st.markdown(f"{icon} **{confidence}**")
        
        #------------------------------------------------------------------------------------
        # Medium Confidence Findings
        #------------------------------------------------------------------------------------
        
        if medium_findings:
            st.markdown("### 🟡 Medium Confidence Findings")
            st.markdown(f"*Probability {medium_conf_threshold*100:.0f}-{high_conf_threshold*100:.0f}% - Possibly present*")
            
            for disease, prob in medium_findings[:top_k]:
                col_a, col_b, col_c = st.columns([4, 2, 2])
                with col_a:
                    is_positive, opt_threshold = get_binary_prediction(prob, disease, use_optimized_thresholds)
                    status = "⚠️ CONSIDER" if is_positive else "ℹ️ MONITOR"
                    st.markdown(f"**{disease}** {status}")
                    if use_optimized_thresholds:
                        st.caption(f"Clinical threshold: {opt_threshold*100:.0f}%")
                with col_b:
                    st.metric("Probability", f"{prob*100:.1f}%")
                with col_c:
                    confidence, icon = classify_confidence(prob, thresholds)
                    st.markdown(f"{icon} **{confidence}**")
        
        #------------------------------------------------------------------------------------
        # Low Confidence Findings (Collapsible)
        #------------------------------------------------------------------------------------
        
        if low_findings:
            with st.expander(f"🟢 Low Confidence Findings ({len(low_findings)} detected)"):
                st.markdown(f"*Probability {low_conf_threshold*100:.0f}-{medium_conf_threshold*100:.0f}% - Unlikely but notable*")
                for disease, prob in low_findings:
                    col_a, col_b = st.columns([6, 2])
                    with col_a:
                        st.markdown(f"{disease}")
                    with col_b:
                        st.markdown(f"{prob*100:.1f}%")
        
        #------------------------------------------------------------------------------------
        # Complete Analysis Table (Optional)
        #------------------------------------------------------------------------------------
        
        if show_all_diseases:
            st.divider()
            st.markdown("### 📊 Complete Analysis (All 14 Pathologies)")
            
            # Create comprehensive DataFrame
            df_data = []
            for disease, prob in results:
                is_positive, opt_threshold = get_binary_prediction(prob, disease, use_optimized_thresholds)
                confidence, icon = classify_confidence(prob, thresholds)
                
                df_data.append({
                    'Pathology': disease,
                    'Probability': f"{prob*100:.1f}%",
                    'Confidence': f"{icon} {confidence}",
                    'Decision': '✅ Positive' if is_positive else '❌ Negative',
                    'Threshold': f"{opt_threshold*100:.0f}%" if use_optimized_thresholds else "50.0%",
                    '_sort': prob
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('_sort', ascending=False).drop('_sort', axis=1)
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
        else:
            # Show top predictions only
            st.divider()
            st.markdown("### 📊 Top Predictions")
            
            top_results = results[:top_k]
            for i, (disease, prob) in enumerate(top_results, 1):
                col_a, col_b = st.columns([7, 3])
                with col_a:
                    st.markdown(f"**#{i}. {disease}**")
                with col_b:
                    st.progress(prob, text=f"{prob*100:.1f}%")
        
        #------------------------------------------------------------------------------------
        # Footer Information
        #------------------------------------------------------------------------------------
        
        st.markdown("---")
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Model", model_version)
        with col_info2:
            st.metric("Device", str(device).upper())
        with col_info3:
            st.metric("Architecture", "DenseNet-121")
        
        st.warning("⚠️ **Disclaimer**: This tool is meant to help and speed up the radiology process but not for complete dependency. All findings must be interpreted by qualified healthcare professionals.")
        
    except Exception as e:
        st.error(f"❌ Inference failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        raise