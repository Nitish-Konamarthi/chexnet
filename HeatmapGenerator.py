import os
import numpy as np
import sys
from PIL import Image

import cv2
import torch
import torchvision.transforms as transforms

# Handle both relative and absolute imports
try:
    from .DensenetModels import DenseNet121, DenseNet169, DenseNet201
except ImportError:
    from DensenetModels import DenseNet121, DenseNet169, DenseNet201

#------------------------------------------------------------------------------------
# Grad-CAM Utilities for Model Interpretability
#------------------------------------------------------------------------------------

def generate_gradcam(model, inp, device, image_rgb=None, blend_alpha=0.5):
    """
    Generate Grad-CAM (Gradient-weighted Class Activation Map) visualization.
    
    This is a reusable function for both web apps and batch processing.
    Uses gradient information to show which regions contributed most to predictions.
    
    Args:
        model: DenseNet121 model (loaded and in eval mode)
        inp: Input tensor (1, 3, 224, 224) already preprocessed and on device
        device: torch device (cuda or cpu)
        image_rgb: Original PIL Image (RGB) for blending (optional)
        blend_alpha: Blend intensity (0.0=all X-ray, 1.0=all heatmap, default=0.5)
    
    Returns:
        numpy array: Grad-CAM heatmap (224, 224, 3) in RGB format, or None if failed
        
    Example:
        >>> from HeatmapGenerator import generate_gradcam
        >>> heatmap = generate_gradcam(model, inp, device, image_rgb, blend_alpha=0.6)
        >>> st.image(heatmap, caption="Grad-CAM Attention Map")
    """
    try:
        # Step 1: Forward pass to identify top prediction
        with torch.no_grad():
            output = model(inp)
            probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
            top_class = np.argmax(probs)
        
        # Step 2: Enable gradients for input
        inp_grad = inp.clone().detach().to(device)
        inp_grad.requires_grad_(True)
        
        output_grad = model(inp_grad)
        score = output_grad[0, top_class]
        
        # Step 3: Backward pass to compute gradients
        model.zero_grad()
        score.backward()
        
        # Step 4: Extract and process gradients
        grads = inp_grad.grad.data  # Shape: (1, 3, 224, 224)
        
        # Compute absolute value and mean across channels
        gradcam = torch.abs(grads).mean(dim=1).squeeze(0)  # Shape: (224, 224)
        
        # Convert to numpy
        gradcam_np = gradcam.detach().cpu().numpy().astype(np.float64)
        
        # Normalize to 0-1 range
        gradcam_min = np.min(gradcam_np)
        gradcam_max = np.max(gradcam_np)
        if gradcam_max > gradcam_min:
            gradcam_normalized = (gradcam_np - gradcam_min) / (gradcam_max - gradcam_min)
        else:
            gradcam_normalized = np.zeros_like(gradcam_np)
        
        # Convert to uint8 for colormap
        gradcam_uint8 = np.uint8(np.round(255 * gradcam_normalized))
        
        # Ensure correct shape and dtype
        if len(gradcam_uint8.shape) != 2:
            gradcam_uint8 = gradcam_uint8.squeeze()
        if gradcam_uint8.dtype != np.uint8:
            gradcam_uint8 = gradcam_uint8.astype(np.uint8)
        
        # Apply colormap (JET: red=high importance, blue=low)
        heatmap_colored = cv2.applyColorMap(gradcam_uint8, cv2.COLORMAP_JET)
        
        # If original image provided, blend them
        if image_rgb is not None:
            img_np = np.array(image_rgb.resize((224, 224))).astype(np.uint8)
            if len(img_np.shape) == 2:  # Grayscale
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            elif img_np.shape[2] == 3:  # RGB to BGR for OpenCV
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Blend images using blend_alpha parameter
            # blend_alpha: 0.0 = 100% X-ray, 0.5 = 50/50, 1.0 = 100% heatmap
            original_weight = 1.0 - blend_alpha
            blended = cv2.addWeighted(img_np, original_weight, heatmap_colored, blend_alpha, 0)
            blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            return blended_rgb
        else:
            # Return heatmap only (convert BGR to RGB)
            heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            return heatmap_rgb
            
    except Exception as e:
        print(f"Grad-CAM generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


#------------------------------------------------------------------------------------
# Legacy HeatmapGenerator Class for Batch Processing
#------------------------------------------------------------------------------------

class HeatmapGenerator:
    """
    Class for batch heatmap generation and file saving.
    
    For real-time web applications, use generate_gradcam() function directly.
    This class provides backward compatibility and convenience methods for batch processing.
    
    Example:
        >>> h = HeatmapGenerator("model.pth", "DENSE-NET-121", 14, 224)
        >>> h.generate("input.png", "output_heatmap.png")
    """
    
    def __init__(self, pathModel, nnArchitecture, nnClassCount, transCrop):
        """
        Initialize HeatmapGenerator.
        
        Args:
            pathModel: Path to model checkpoint
            nnArchitecture: Model architecture ('DENSE-NET-121', 'DENSE-NET-169', 'DENSE-NET-201')
            nnClassCount: Number of output classes (14 for CheXNet)
            transCrop: Image size after preprocessing (224)
        """
       
        # Initialize the network
        if nnArchitecture == 'DENSE-NET-121': 
            model = DenseNet121(nnClassCount, True)
        elif nnArchitecture == 'DENSE-NET-169': 
            model = DenseNet169(nnClassCount, True)
        elif nnArchitecture == 'DENSE-NET-201': 
            model = DenseNet201(nnClassCount, True)
        else:
            raise ValueError(f"Unknown architecture: {nnArchitecture}")
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(self.device)

        # Load checkpoint
        if not os.path.exists(pathModel):
            raise FileNotFoundError(f"Model checkpoint not found at {pathModel}")
        
        modelCheckpoint = torch.load(pathModel, map_location=self.device)
        state_dict = modelCheckpoint.get('state_dict', modelCheckpoint)
        
        # Remove DataParallel wrapper if present
        new_state = {}
        for k, v in state_dict.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_state[nk] = v
        
        model.load_state_dict(new_state, strict=False)
        self.model = model
        self.model.eval()
        
        # Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transformSequence = transforms.Compose([
            transforms.Resize(transCrop),
            transforms.ToTensor(),
            normalize
        ])
        self.transCrop = transCrop
    
    def generate(self, pathImageFile, pathOutputFile, transCrop=None):
        """
        Generate Grad-CAM heatmap and save to file.
        
        Args:
            pathImageFile: Path to input image
            pathOutputFile: Path to save output heatmap image
            transCrop: Image crop size (uses self.transCrop if not specified)
            
        Returns:
            numpy array: Generated heatmap image
        """
        
        if transCrop is None:
            transCrop = self.transCrop
        
        # Load and preprocess image
        imageData = Image.open(pathImageFile).convert('RGB')
        image_tensor = self.transformSequence(imageData)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Generate Grad-CAM using shared function
        heatmap_img = generate_gradcam(self.model, image_tensor, self.device, imageData)
        
        if heatmap_img is None:
            raise RuntimeError("Failed to generate Grad-CAM")
        
        # Save output
        output_dir = os.path.dirname(pathOutputFile)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert RGB to BGR for cv2.imwrite
        heatmap_bgr = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(pathOutputFile, heatmap_bgr)
        print(f"✓ Heatmap saved to {pathOutputFile}")
        
        return heatmap_img


#------------------------------------------------------------------------------------
# Command-line Interface for Batch Processing
#------------------------------------------------------------------------------------

if __name__ == "__main__":
    pathInputImage = 'test/00009285_000.png'
    pathOutputImage = 'test/heatmap.png'
    pathModel = 'chexnet/models/m-25012018-123527.pth.tar'

    nnArchitecture = 'DENSE-NET-121'
    nnClassCount = 14
    transCrop = 224

    try:
        h = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)
        h.generate(pathInputImage, pathOutputImage, transCrop)
        print("✓ Batch heatmap generation completed successfully!")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
