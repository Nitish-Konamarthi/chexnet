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

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import the DenseNet class from the cloned repo; adjust path if needed
try:
    from DensenetModels import DenseNet121
    from HeatmapGenerator import generate_gradcam, HeatmapGenerator
except Exception as e:
    st.error(f"Could not import modules: {str(e)}")
    raise

st.set_page_config(page_title="CheXNet Prototype", layout="wide")

MODEL_PATH_ENV = os.environ.get("MODEL_PATH", "chexnet/models/m-25012018-123527.pth.tar")
st.title("🩺 Chest X-ray Analysis")

st.markdown(
    """
    ** DenseNet121-based chest X-ray pathology detection** with interpretable heatmap visualization.
    
    """
)

# Sidebar: model settings and configuration
with st.sidebar:
    st.header("⚙️ Settings & Configuration")
    
    model_path = st.text_input("Model path", value=MODEL_PATH_ENV)
    top_k = st.number_input("Top K predictions", min_value=1, max_value=14, value=7)
    
    st.divider()
    st.subheader("Visualization Options")
    show_heatmap = st.checkbox("Show Class Activation Map (CAM)", value=True)
    heatmap_blend_alpha = st.slider("Heatmap blend intensity", 0.0, 1.0, 0.5)
    
    st.divider()
    st.subheader("Inference Settings")
    confidence_threshold = st.slider("Confidence threshold (%)", 0, 100, 10)
    
    run_inference = st.button("🔄 (Re)load Model")

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
    """Load HeatmapGenerator for CAM visualization."""
    try:
        h = HeatmapGenerator(path, "DENSE-NET-121", 14, 224)
        return h
    except Exception as e:
        st.warning(f"Could not load HeatmapGenerator: {e}. CAM visualization will be disabled.")
        return None

# Preprocessing (match training-ish transforms)
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.485,0.485],[0.229,0.229,0.229])
])

CLASS_NAMES = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule',
               'Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema',
               'Fibrosis','Pleural_Thickening','Hernia']

# Load model (cached)
try:
    model, model_version, device = load_model(model_path)
    st.sidebar.success(f"✓ Loaded {model_version}\nDevice: {device}")
except Exception as e:
    st.sidebar.error(f"Model load failed: {str(e)}")
    st.stop()

# Optionally load heatmap generator
heatmap_generator = None
if show_heatmap:
    heatmap_generator = load_heatmap_generator(model_path)

# Image uploader
st.subheader("📤 Upload Chest X-ray Image")
uploaded = st.file_uploader("Select a PNG/JPG/TIFF chest X-ray image", type=["png","jpg","jpeg","tiff"])
if uploaded is None:
    st.info("👆 Upload an X-ray image to begin analysis.")
    st.stop()

# Read and prepare image
image_bytes = uploaded.read()
image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
image_rgb = Image.open(io.BytesIO(image_bytes)).convert("RGB")

# Display uploaded image in columns
col1, col2 = st.columns(2)
with col1:
    st.image(image, caption="Uploaded X-ray (grayscale)", use_column_width=True)

# Prediction function
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

# Run prediction button
if st.button(" Run Pathology Analysis", key="run_pred", use_container_width=True):
    try:
        # Get predictions
        results, inp = predict_pil(image, model, device)
        
        # Generate Grad-CAM if enabled
        cam_img = None
        if show_heatmap and heatmap_generator is not None:
            try:
                with st.spinner("Generating Grad-CAM attention map..."):
                    cam_img = generate_gradcam_visualization(image_rgb, model, inp, device, heatmap_blend_alpha)
            except Exception as e:
                st.warning(f"Could not generate Grad-CAM: {e}")
        
        # Display Grad-CAM heatmap in column 2 if available
        if cam_img is not None:
            with col2:
                st.image(cam_img, caption="Grad-CAM: Model Attention Regions", use_column_width=True)
        
        st.divider()
        
        # Display results
        st.subheader("🔍 Pathology Predictions")
        
        # Top predictions with visual indicators
        filtered_results = [(label, prob) for label, prob in results if prob * 100 >= confidence_threshold]
        
        if filtered_results:
            for i, (label, prob) in enumerate(filtered_results[:top_k], 1):
                prob_pct = prob * 100
                col_a, col_b = st.columns([2, 8])
                with col_a:
                    st.metric(f"#{i}", label, f"{prob_pct:.1f}%")
                with col_b:
                    st.progress(prob, text=f"{prob_pct:.1f}%")
        else:
            st.info(f"No pathologies detected above {confidence_threshold}% threshold.")
        
        # Show all probabilities in a detailed table
        st.subheader("📊 Complete Analysis")
        import pandas as pd
        df = pd.DataFrame(results, columns=["Pathology", "Probability"])
        df["Confidence (%)"] = (df["Probability"] * 100).round(2)
        df = df.sort_values("Confidence (%)", ascending=False)
        
        st.dataframe(
            df[["Pathology", "Confidence (%)"]].rename(columns={"Confidence (%)": "Confidence %"}),
            use_container_width=True,
            hide_index=True
        )
        
        # Footer info
        st.markdown("---")
        st.caption(f"Model: {model_version} | Device: {device} | Architecture: DenseNet121 | Classes: 14")
        st.warning("Disclaimer: This tool is meant to help and faster the radialogy process but not for complete Dependency.")
        
    except Exception as e:
        st.error(f"Inference failed: {e}")
        raise
