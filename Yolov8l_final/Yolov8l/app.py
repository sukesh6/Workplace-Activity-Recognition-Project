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
def predict(model, image_or_video):
    results = model(image_or_video)
    return results

# Save output image with predictions
def save_and_display_image_results(results, uploaded_image_path):
    output_image = results[0].plot()  # Get the image with bounding boxes
    save_path = "output_image.jpg"  # Save the result
    cv2.imwrite(save_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    return save_path

# Process video and save the output
def process_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output video writer
    output_video_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    st.write("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict on each frame
        results = predict(model, frame)
        output_frame = results[0].plot()

        # Write the processed frame to the output video
        out.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()
    return output_video_path

# Streamlit app
st.title("YOLOv8 Object Detection")

# Sidebar for uploading images or videos
uploaded_file = st.sidebar.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Model path (update this to the correct path for your weights file)
model_weights_path = "C:/Users/sukes/OneDrive/Desktop/360/Project ( Workplace Activity Recognition System)/Yolo/yolov8s_final/best.pt"

# Load model
model = load_model(model_weights_path)

if uploaded_file:
    file_type = uploaded_file.type

    if "image" in file_type:
        # Read and display the uploaded image
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Uploaded Image")

        # Convert image to OpenCV format for YOLOv8 processing
        image_array = np.array(uploaded_image)

        # Predict and process results
        st.write("Running inference on the image...")
        results = predict(model, image_array)

        # Display results
        st.write("Detection completed. Visualizing results...")
        output_image_path = save_and_display_image_results(results, uploaded_file.name)
        st.image(output_image_path, caption="Detection Results")

        # Display raw results
        st.write("Detection Details:")
        st.json(results[0].boxes.xyxy.tolist())  # List of bounding boxes with confidence scores

    elif "video" in file_type:
        # Save uploaded video to a file
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process video
        output_video_path = process_video(model, video_path)

        # Display the processed video
        st.write("Detection completed. Displaying video results...")
        st.video(output_video_path)

else:
    st.info("Please upload an image or video to get started.")
