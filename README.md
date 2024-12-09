# Monocular Depth Estimation

## 📜 **Project Description**
This project demonstrates monocular depth estimation using three different models: MiDaS, Depth Anything V1, and Depth Anything V2. The application allows users to upload an image and visualize the depth maps in grayscale and color. It is implemented using Python and Streamlit for the frontend, while leveraging OpenCV, PyTorch, and other libraries for image processing and inference.

## 📂 **Features**  
- **Multi-Model Support:** Integrates three depth estimation models for comparison and analysis.
- **Interactive Frontend:** A Streamlit-based web app for easy image upload and result visualization.
- **Depth Map Visualization:** Outputs grayscale and colored depth maps.
- **Dynamic Inference:** Processes images in real time using selected models.
---


## 🚀 **Getting Started**

### **Prerequisites**   
* Ensure the following libraries are installed with the specified versions: 

    ```bash
        opencv-python==4.10.0.84
        numpy==1.26.4
        matplotlib==3.9.3
        torch==2.5.1
        streamlit==1.40.2
        
### **Installation**
Follow these steps to set up the project locally:

1. Clone the repository:
    ```bash
    git clone git@github.com:MitPatel24/Monocular_Depth_Map.git
    cd <working-directory>

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

### 📋 **Usage**
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
2. Open the app in your browser.
3. Upload an image in JPG/PNG format.
4. Select a depth estimation model from the sidebar.
5. View the original image, grayscale depth map, and colored depth map.

## **Contributors**
- Mitkumar Patel -  [MitPatel24](https://github.com/MitPatel24)









