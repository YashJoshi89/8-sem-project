import cv2
import numpy as np

def detect_pe_location(image_array):
    """
    Detects Pulmonary Embolism (PE) location in a DICOM image.
    Returns processed image with bounding box and estimated PE location.
    """
    # Convert image to grayscale (if not already)
    image = (image_array / np.max(image_array) * 255).astype(np.uint8)
    
    # Apply Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 30, 100)

    # Find contours (possible clot regions)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define PE location
    pe_location = "Unknown"
    height, width = image.shape

    # Draw Bounding Box Around Largest Contour (if any)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue Box

        # Estimate PE Location
        if x + w / 2 < width / 2:
            pe_location = "Left Lung"
        else:
            pe_location = "Right Lung"

    return image, pe_location
