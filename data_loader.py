import os
import pydicom
import numpy as np

def load_dicom_data(image_dir):
    images = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".dcm"):
                dicom_path = os.path.join(root, file)
                try:
                    dicom = pydicom.dcmread(dicom_path)
                    img_array = dicom.pixel_array.astype(np.float32)
                    img_array /= np.max(img_array)  
                    images.append(np.expand_dims(img_array, axis=-1))
                except Exception as e:
                    print(f"❌ Error reading {dicom_path}: {e}")
    if len(images) == 0:
        raise ValueError("No DICOM images found!")
    return np.array(images, dtype=np.float32)
