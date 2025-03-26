# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:37:59 2025

@author: Keerati
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Define paths
hf_path = 'D:/Waveforms/HF_seg/'
normal_path = 'D:/Waveforms/Normal_seg/'

# Load and label data
def load_data(path, label, segment_size=1000):
    data = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        # Select only Channel A and Channel B columns
        df = df[['Channel A', 'Channel B']]
        # Convert to numeric, coerce invalid values to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        # Drop rows with NaN values
        df = df.dropna()
        # Check if the dataframe has enough rows
        if len(df) >= segment_size:
            # Take only the first segment_size rows
            df = df.iloc[:segment_size].values
            data.append((df, label))
    return data

# Load HF and Normal data
hf_data = load_data(hf_path, 1)
normal_data = load_data(normal_path, 0)

# Combine data
all_data = hf_data + normal_data
np.random.shuffle(all_data)  # Shuffle data

# Split features and labels
X = np.array([sample[0] for sample in all_data])  # Features (Channel A and B)
y = np.array([sample[1] for sample in all_data])  # Labels (0 or 1)

# Normalize the data
X = X.astype(float)  # Ensure X is float type
X = X / np.max(X)

# Convert labels to categorical
y = to_categorical(y, num_classes=2)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the 1D CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the best weights
checkpoint = ModelCheckpoint('best_model_1DCNN_20-1-25.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train the model
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[checkpoint]
)

# Load the best weights
model.load_weights('best_model_1DCNN_20-1-25.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Predictions and Metrics
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(y_true, y_pred_classes, target_names=["Normal", "HF"])
print("Classification Report:\n", class_report)

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])  # Use probabilities for the positive class
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()
