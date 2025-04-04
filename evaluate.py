from data_loader import load_dicom_data
from model import PEModel

def evaluate_model():
    model_path = "./pe_detection_model"
    test_dir = "./CT/test"

    X_test = load_dicom_data(test_dir)

    model = PEModel()
    model.load_weights(model_path)
    print("✅ Model loaded for evaluation.")

    predictions = model(X_test)
    print("🧾 Predictions:", predictions.numpy())
