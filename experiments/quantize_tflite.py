"""Experimental TFLite float16 conversion (not on the Docker serve path).

Hardcoded SavedModel path is historical; update before running.
"""
import tensorflow as tf

# Set the path to your SavedModel directory
saved_model_dir = "models/model_20250510_213840/"

# Create the converter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Enable float16 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert the model to TFLite format
tflite_quant_model = converter.convert()

# Save the TFLite model
with open("model_quant_float16.tflite", "wb") as f:
    f.write(tflite_quant_model)

