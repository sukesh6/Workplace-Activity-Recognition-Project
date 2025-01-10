

# Place set_page_config at the top
import streamlit as st

st.set_page_config(page_title="YOLO8 Video Detection", page_icon="🎥", layout="centered")

# Import other libraries only after set_page_config
from ultralytics import YOLO
import cv2
import tempfile
import os

st.title("Video Object Detection with YOLOv8s")
st.markdown("Upload a video and let the YOLOv8s model detect objects in it.")

# Video uploader
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    video_path = temp_file.name

    # Load the YOLO model
    model = YOLO(r"best.pt")

    # Process the video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20

    # Temporary output file
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_file.name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stframe = st.empty()  # Placeholder for video frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the frame
        results = model(frame)

        # Draw detections on the frame
        annotated_frame = results[0].plot()

        # Write the frame to the output file
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        # Display the frame in Streamlit
        stframe.image(annotated_frame, channels="BGR")

    cap.release()
    out.release()

    # Provide download link for the processed video
    with open(output_path, "rb") as file:
        st.download_button(
            label="Download Processed Video",
            data=file,
            file_name="output.mp4",
            mime="video/mp4"
        )

    # Cleanup temporary files
    os.remove(video_path)
    os.remove(output_path)
