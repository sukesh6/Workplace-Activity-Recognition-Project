import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Load YOLOv8 model
@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)

# Perform prediction
def predict(model, image):
    results = model(image)
    return results

# Save output image with predictions
def save_and_display_results(results, uploaded_image_path):
    output_image = results[0].plot()  # Get the image with bounding boxes
    save_path = "output_image.jpg"  # Save the result
    cv2.imwrite(save_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    return save_path

# Streamlit app
st.title("YOLOv8 Object Detection")

# Sidebar for uploading images
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Model path (update this to the correct path for your weights file)
model_weights_path = "C:/Users/sukes/OneDrive/Desktop/360/Project ( Workplace Activity Recognition System)/Yolo/Yolov8l_final/Yolov8l/best (1).pt"

# Load model
model = load_model(model_weights_path)

if uploaded_file:
    # Read and display the uploaded image
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image")  # Updated parameter

    # Convert image to OpenCV format for YOLOv8 processing
    image_array = np.array(uploaded_image)

    # Predict and process results
    st.write("Running inference on the image...")
    results = predict(model, image_array)

    # Display results
    st.write("Detection completed. Visualizing results...")
    output_image_path = save_and_display_results(results, uploaded_file.name)
    st.image(output_image_path, caption="Detection Results")  # Updated parameter

    # Display raw results
    st.write("Detection Details:")
    st.json(results[0].boxes.xyxy.tolist())  # List of bounding boxes with confidence scores
else:
    st.info("Please upload an image to get started.")