Early Detection of Pulmonary Embolism
This project is focused on the early detection of Pulmonary Embolism (PE) from CT scan DICOM images using deep learning. It provides a simple CNN-based model, a graphical user interface (GUI), and visualization tools for PE location identification.

Project Features
Load and preprocess DICOM images

Train a CNN-based PE classification model

Detect and visualize suspected PE locations

Interactive GUI for medical image upload and prediction

Directory Structure
. ├── data_loader.py # DICOM image loading and preprocessing
├── model.py # CNN model definition using TensorFlow
├── train.py # Model training script
├── evaluate.py # Evaluation script
├── predict.py # Run prediction on a single image
├── gui.py # GUI for uploading and analyzing images
├── location_detector.py # PE location detection using OpenCV
├── CT/ # DICOM dataset folder (train/test)
└── pe_detection_model/ # Folder to store model weights

Installation
Install required Python libraries:

pip install tensorflow pydicom numpy opencv-python pillow

Usage
Train the model: python train.py

Evaluate model performance: python evaluate.py

Predict PE on a single image: python predict.py

Run the graphical user interface: python gui.py

Model & Training Notes
The model is a basic CNN. Replace dummy labels with real labels for accurate training.

Training is currently slow and not optimal.

To improve:

Use a pre-trained model like MobileNetV2 (transfer learning)

Add BatchNormalization, Dropout, and more layers

Resize all images to a fixed shape (e.g., 128x128)

Use tf.data API with caching and prefetching

Replace manual training loop with model.fit()

Future Improvements
Add Grad-CAM or other visualization tools for interpretability

Integrate 3D model support for volumetric CT scans

Expand GUI with batch processing and reporting