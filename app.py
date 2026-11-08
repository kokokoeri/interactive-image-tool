import sys
import types
import os
import importlib.util

mock_pkg = types.ModuleType('pkg_resources')

class MockRequirement:
    @staticmethod
    def parse(name):
        return name

def mock_resource_filename(req, resource_name):
    spec = importlib.util.find_spec('cvlib')
    if spec and spec.submodule_search_locations:
        pkg_path = spec.submodule_search_locations[0]
        base_dir = os.path.dirname(pkg_path)
        full_path = os.path.join(base_dir, resource_name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return full_path
    return resource_name

mock_pkg.Requirement = MockRequirement
mock_pkg.resource_filename = mock_resource_filename
sys.modules['pkg_resources'] = mock_pkg

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import math
import cvlib as cv

def get_histogram_df(channel_data, col_name):
    hist = cv2.calcHist([channel_data], [0], None, [256], [0, 256])
    return pd.DataFrame(hist, columns=[col_name])

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def main():
    st.set_page_config(layout="wide")
    st.title("Image Processing and Face Analysis Tool")
    
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        
        st.write("Resolution:", f"{img_array.shape[1]}x{img_array.shape[0]}")
        st.write("Color Depth:", img_array.shape[2])
        st.write("File Size:", f"{uploaded_file.size / 1024:.2f} KB")
        
        mode = st.radio("Select View Mode", ("Original", "RGB Channels", "YUV", "Spatial Filtering", "Face Detection"))
        
        if mode == "Original":
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                df = get_histogram_df(gray, "Grayscale")
                st.line_chart(df)
                
        elif mode == "RGB Channels":
            r, g, b = cv2.split(img_array)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(r, caption="Red Channel", use_container_width=True)
                st.line_chart(get_histogram_df(r, "Red"), color="#FF0000")
            with col2:
                st.image(g, caption="Green Channel", use_container_width=True)
                st.line_chart(get_histogram_df(g, "Green"), color="#00FF00")
            with col3:
                st.image(b, caption="Blue Channel", use_container_width=True)
                st.line_chart(get_histogram_df(b, "Blue"), color="#0000FF")
                
        elif mode == "YUV":
            yuv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
            y, u, v = cv2.split(yuv_img)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(y, caption="Y (Luminance)", use_container_width=True)
                st.line_chart(get_histogram_df(y, "Y"))
            with col2:
                st.image(u, caption="U (Chrominance)", use_container_width=True)
                st.line_chart(get_histogram_df(u, "U"))
            with col3:
                st.image(v, caption="V (Chrominance)", use_container_width=True)
                st.line_chart(get_histogram_df(v, "V"))
                
        elif mode == "Spatial Filtering":
            st.subheader("Custom Kernel Spatial Filtering")
            
            default_kernel = "0, -1, 0\n-1, 5, -1\n0, -1, 0"
            kernel_input = st.text_area("Enter Kernel Matrix (comma separated, new line for each row)", value=default_kernel)
            normalize = st.checkbox("Apply Normalization")
            
            try:
                rows = kernel_input.strip().split('\n')
                kernel_list = [[float(val.strip()) for val in row.split(',')] for row in rows]
                kernel = np.array(kernel_list)
                
                if kernel.shape[0] != kernel.shape[1]:
                    st.error("錯誤：卷積核必須是方陣 (如 3x3 或 5x5)！")
                else:
                    if normalize:
                        kernel_sum = np.sum(kernel)
                        if kernel_sum != 0:
                            kernel = kernel / kernel_sum
                        else:
                            st.warning("警告：矩陣總和為 0，已略過正規化以避免除以零錯誤。")
                    
                    st.write("Current Kernel:")
                    st.dataframe(kernel)
                    
                    filtered_img = cv2.filter2D(img_array, -1, kernel)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(filtered_img, caption="Filtered Image", use_container_width=True)
                    
                    psnr_val = calculate_psnr(img_array, filtered_img)
                    st.success(f"**PSNR (Peak Signal-to-Noise Ratio):** {psnr_val:.2f} dB")
                    st.latex(r"PSNR = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{MSE} \right)")
                    
            except ValueError:
                st.error("輸入格式錯誤：請確保輸入的都是數字，且同一列以逗號分隔。")
                
        elif mode == "Face Detection":
            st.subheader("Face Detection (Deep Learning - cvlib)")
            
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
            
            faces, confidences = cv.detect_face(img_array, threshold=confidence_threshold)
            
            output_image = img_array.copy()
            face_count = len(faces)
            
            if face_count > 0:
                for i, (face, conf) in enumerate(zip(faces, confidences)):
                    startX, startY, endX, endY = face
                    cv2.rectangle(output_image, (startX, startY), (endX, endY), (0, 255, 0), 3)
                    text = f"Face {i+1}: {conf:.2f}"
                    cv2.putText(output_image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    st.write(f"**人臉 {i+1} 信心度 (Confidence Score):** {conf*100:.2f}%")
            
            st.success(f"**總共偵測到的人臉數量:** {face_count}")
            st.image(output_image, caption="Face Detection Result", use_container_width=True)

if __name__ == "__main__":
    main()