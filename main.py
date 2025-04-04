import os
import tensorflow as tf
from data_loader import load_dicom_data
from model import PEModel
from evaluate import evaluate_model
from predict import predict_on_image

# Paths
image_dir = "./CT/train"
test_image_path = "./CT/test/D0210.dcm"
model_path = "./pe_detection_model"

# Load DICOM Data
X_train = load_dicom_data(image_dir)

# Initialize Model
model = PEModel()

if os.path.exists(model_path + ".index"):
    model.load_weights(model_path)
    print("✅ Model weights loaded successfully.")
else:
    print("🚀 Training a new model...")

# Training Parameters
optimizer = tf.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.losses.BinaryCrossentropy()

# Dummy Labels (Replace with actual labels if available)
y_train = tf.zeros((X_train.shape[0], 1), dtype=tf.float32)

# Prepare Dataset
batch_size = 16
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Training Loop
for epoch in range(3):
    epoch_loss = 0.0
    for batch_X, batch_y in dataset:
        with tf.GradientTape() as tape:
            predictions = model(batch_X)
            loss = loss_fn(batch_y, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        if gradients and all(g is not None for g in gradients):
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        epoch_loss += loss.numpy()

    print(f"✅ Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

# Save Model
model.save_weights(model_path)
print(f"✅ Model saved to {model_path}")

# Evaluate & Predict
evaluate_model()
print("🔍 Prediction on new image:", predict_on_image(test_image_path))
