import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import shutil

class DepthEstimationApp:
    def __init__(self):
        # Initialize models
        self.initialize_models()
    
    def initialize_models(self):
        # MiDaS Model
        try:
            self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self.midas.to('cpu')
            self.midas.eval()
            
            # MiDaS Transforms
            self.midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            self.midas_transform = self.midas_transforms.small_transform
        except Exception as e:
            st.error(f"Error loading MiDaS model: {e}")
            self.midas = None

    def process_midas(self, img):
        """Process image using MiDaS model"""
        if self.midas is None:
            st.error("MiDaS model not loaded")
            return None, None
        
        # Convert and prepare image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_batch = self.midas_transform(img_rgb).to('cpu')
        
        # Predict depth
        with torch.no_grad():
            prediction = self.midas(img_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
            
            output = prediction.cpu().numpy()
            
            # Normalize for visualization
            output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            output_norm = (output_norm * 255).astype(np.uint8)
            
            # Grayscale depth map
            depth_gray = output_norm
            
            # Colored depth map (Magma/Inferno)
            depth_color = cv2.applyColorMap(output_norm, cv2.COLORMAP_INFERNO)
        
        return depth_gray, depth_color
    
    def process_depth_anything_v1(self, img_path):
        """Process image using Depth Anything V1 with manual model loading"""
        try:
            # Ensure repository is cloned
            if not os.path.exists('Depth-Anything'):
                st.info("Cloning Depth Anything V1 repository...")
                subprocess.run(['!git', 'clone', 'https://github.com/LiheYoung/Depth-Anything'], check=True)
            
            # Manual dependency installation
            st.info("Installing manual dependencies...")
            subprocess.run([
                'pip', 'install', '--user', '--upgrade', 'torch', 'torchvision', 'huggingface_hub', 
                'transformers', 'opencv-python', 'numpy', 'matplotlib'
            ], check=True)
            
            # Add repository to Python path
            repo_path = os.path.abspath('Depth-Anything')
            sys.path.append(repo_path)
            
            # Import required modules
            from depth_anything.dpt import DepthAnything
            import torch
            
            # Determine device
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize model dynamically
            st.info("Initializing Depth Anything V1 model...")
            depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vits14').to(DEVICE).eval()
            
            # Perform inference
            with torch.no_grad():
                depth = depth_anything.infer(img_path)
            
            # Normalize and process depth
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_gray = depth_norm.astype('uint8')
            depth_color = cv2.applyColorMap(depth_norm.astype('uint8'), cv2.COLORMAP_INFERNO)
            
            return depth_gray, depth_color
            
        except Exception as e:
            st.error(f"Detailed V1 processing error: {e}")
            return None, None

    def process_depth_anything_v2(self, img):
        """Enhanced Depth Anything V2 processing with image resizing"""
        try:
            # Ensure repository is cloned
            repo_path = 'Depth-Anything-V2'
            if not os.path.exists(repo_path):
                st.info("Cloning Depth Anything V2 repository...")
                subprocess.run(['git', 'clone', 'https://huggingface.co/spaces/depth-anything/Depth-Anything-V2'], check=True)
            
            # Add repository to Python path
            sys.path.append(os.path.abspath(repo_path))
            
            # Import model dynamically
            from depth_anything_v2.dpt import DepthAnythingV2
            import torch
            import torchvision.transforms as transforms
            
            # Image preprocessing with resizing
            def prepare_image(img):
                # Calculate new dimensions (multiples of 14)
                h, w = img.shape[:2]
                new_h = (h // 14) * 14
                new_w = (w // 14) * 14

                # Resize image
                resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Convert and normalize
                img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)
                
                return img_tensor, resized_img

            # Determine device
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize model
            st.info("Initializing Depth Anything V2 model...")
            model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384]).to(DEVICE)
            
            # Prepare image
            img_tensor, resized_img = prepare_image(img)
            img_tensor = img_tensor.to(DEVICE)
            
            # Inference
            with torch.no_grad():
                depth = model(img_tensor)
                depth = depth.squeeze().cpu().numpy()
            
            # Resize depth map back to original image size
            depth_resized = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Depth map processing
            depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
            depth_gray = depth_norm.astype('uint8')
            # depth_color = cv2.applyColorMap(depth_norm.astype('uint8'), cv2.COLORMAP_INFERNO)
            depth_color=cv2.applyColorMap((img*255).astype(np.uint8),cv2.COLORMAP_INFERNO)

            return depth_gray, depth_color

        except Exception as e:
            st.error(f"Detailed V2 processing error: {e}")
            return None, None

    def run(self):
        """Streamlit App Main Function"""
        st.title('Monocular Depth Estimation')
        
        # Sidebar for model selection
        st.sidebar.header('Model Selection')
        model_choice = st.sidebar.selectbox(
            'Choose Depth Estimation Model',
            ['MiDaS', 'Depth Anything V1', 'Depth Anything V2']
        )
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Read the image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original image
            st.subheader('Original Image')
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Process based on selected model
            try:
                if model_choice == 'MiDaS':
                    depth_gray, depth_color = self.process_midas(img)
                elif model_choice == 'Depth Anything V1':
                    # Save uploaded file temporarily
                    temp_path = 'temp_input.jpg'
                    cv2.imwrite(temp_path, img)
                    depth_gray, depth_color = self.process_depth_anything_v1(temp_path)
                    os.remove(temp_path)
                else:  # Depth Anything V2
                    depth_gray, depth_color = self.process_depth_anything_v2(img)
                
                # Display results
                if depth_gray is not None and depth_color is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader('Grayscale Depth Map')
                        st.image(depth_gray, clamp=True, use_container_width=True)
                    
                    with col2:
                        st.subheader('Colored Depth Map')
                        st.image(depth_color, clamp=True, use_container_width=True)
                else:
                    st.error("Failed to generate depth map")
            
            except Exception as e:
                st.error(f"Unexpected error during depth estimation: {e}")

def main():
    app = DepthEstimationApp()
    app.run()

if __name__ == "__main__":
    main()