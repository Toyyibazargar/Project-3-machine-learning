# Project-3-machine-learning

Handwritten Digit Prediction - Classification Analysis

Objective
To classify handwritten digits (0-9) using machine learning models by analyzing and predicting based on image data.

Data Source
The MNIST dataset is commonly used for handwritten digit classification. It contains 70,000 grayscale images (28x28 pixels) of handwritten digits (60,000 for training and 10,000 for testing).

Source: Kaggle MNIST Dataset
Import Library
python
Copy code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
Import Data
python
Copy code
# Load the MNIST dataset (CSV format for simplicity)
data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")
Describe Data
python
Copy code
# Display the structure and statistics of the data
print(data.head())  # View the first few rows
print(data.info())  # Check data types and missing values
print(data.describe())  # Statistical summary
Dataset Details:
The first column represents the target variable (digits 0-9).
Remaining columns (784) represent pixel intensities (28x28 images flattened into 1D arrays).
Data Visualization
python
Copy code
# Visualize some digits from the dataset
fig, axes = plt.subplots(1, 5, figsize=(10, 4))
for i in range(5):
    digit = data.iloc[i, 1:].values.reshape(28, 28)
    axes[i].imshow(digit, cmap='gray')
    axes[i].set_title(f"Label: {data.iloc[i, 0]}")
    axes[i].axis('off')
plt.show()
Data Preprocessing
python
Copy code
# Separate features and target
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Normalize pixel values (0-255) to range (0-1)
X = X / 255.0

# One-hot encode the target variable for neural network modeling
y = to_categorical(y)

# Repeat for the test dataset
X_test = test_data.iloc[:, 1:].values / 255.0
y_test = to_categorical(test_data.iloc[:, 0].values)
Define Target Variable (y) and Feature Variables (X)
Target Variable (y): The digit label (0-9).
Feature Variables (X): Pixel intensities from the image.
Train Test Split
python
Copy code
# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
Modeling
Random Forest Classifier
python
Copy code
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.argmax(axis=1))  # Use non-one-hot target
Neural Network
python
Copy code
nn_model = Sequential([
    Flatten(input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = nn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
Model Evaluation
Random Forest
python
Copy code
rf_predictions = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test.argmax(axis=1), rf_predictions))
print(confusion_matrix(y_test.argmax(axis=1), rf_predictions))
print(classification_report(y_test.argmax(axis=1), rf_predictions))
Neural Network
python
Copy code
nn_loss, nn_accuracy = nn_model.evaluate(X_test, y_test)
print("Neural Network Accuracy:", nn_accuracy)

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
Prediction
python
Copy code
# Randomly select an image for prediction
sample_image = X_test[0].reshape(1, 784)
plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.title("Sample Image")
plt.show()

# Predict with Neural Network
nn_prediction = nn_model.predict(sample_image)
print("Neural Network Prediction:", np.argmax(nn_prediction))

# Predict with Random Forest
rf_prediction = rf_model.predict(sample_image)
print("Random Forest Prediction:", rf_prediction[0])
Explanation
Data Preprocessing: The image data was normalized to improve model performance.
Modeling Approaches:
Random Forest for simpler baseline classification.
Neural Network for more complex pattern recognition.
Results:
Neural Network typically outperforms Random Forest due to its ability to capture spatial relationships in pixel data.
Visualization: The plotted images and confusion matrix help understand model predictions and errors.
Applications: This analysis can be applied to postal code recognition, check processing, and other OCR tasks.
