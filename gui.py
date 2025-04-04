import tkinter as tk
from tkinter import filedialog, Label
import pydicom
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageTk
from model import PEModel
from location_detector import detect_pe_location  # New function for location detection

# Load Model
model_path = "./pe_detection_model"
model = PEModel()
model.load_weights(model_path)

# GUI Setup
root = tk.Tk()
root.title("Pulmonary Embolism Detection with Location")
root.geometry("600x600")

# Function to Process DICOM File
def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("DICOM Files", "*.dcm")])
    if file_path:
        try:
            # Read DICOM Image
            dicom = pydicom.dcmread(file_path)
            img_array = dicom.pixel_array.astype(np.float32)
            img_array /= np.max(img_array)  # Normalize
            img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
            img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
            
            # Model Prediction
            prediction = model(img_array).numpy()[0][0]
            
            # Determine PE Severity
            if prediction < 0.4:
                severity = "🟢 Mild"
                color = "green"
            elif prediction < 0.7:
                severity = "🟡 Moderate"
                color = "orange"
            else:
                severity = "🔴 Severe"
                color = "red"
            
            # Detect PE Location
            processed_img, pe_location = detect_pe_location(dicom.pixel_array)

            # Convert Processed Image for Display
            processed_img = cv2.resize(processed_img, (200, 200))
            processed_img = Image.fromarray(processed_img)
            processed_tk = ImageTk.PhotoImage(processed_img)

            # Convert Original DICOM Image for Display
            img = Image.fromarray((dicom.pixel_array / np.max(dicom.pixel_array) * 255).astype(np.uint8))
            img = img.resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)

            # Update GUI Elements
            original_label.config(image=img_tk)
            original_label.image = img_tk

            processed_label.config(image=processed_tk)
            processed_label.image = processed_tk

            result_label.config(text=f"PE Intensity: {severity} (Score: {prediction:.4f})\nLocation: {pe_location}", fg=color)
        
        except Exception as e:
            result_label.config(text=f"❌ Error: {e}", fg="red")

# GUI Elements
upload_button = tk.Button(root, text="Upload DICOM File", command=upload_file, font=("Arial", 12))
upload_button.pack(pady=20)

original_label = Label(root, text="Original Image")
original_label.pack()

processed_label = Label(root, text="Processed Image with PE Highlighted")
processed_label.pack()

result_label = Label(root, text="Upload a DICOM file to analyze", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()
