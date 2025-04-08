import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the trained model
model = tf.keras.models.load_model("traffic_sign_classifier.h5")

# Define test directory and class labels
test_dir = "dataset/test"
class_labels = sorted(os.listdir(test_dir))  # Automatically get class names from folder names

# Initialize lists to store true and predicted labels
y_true = []
y_pred = []

# Function to predict and display images
def predict_images(test_dir):
    for label in class_labels:
        label_dir = os.path.join(test_dir, label)
        if not os.path.isdir(label_dir):
            continue  # Skip if not a directory

        print(f"\nTesting images for class: {label}\n")
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)

            # Load and preprocess image
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict class
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions)
            predicted_class = class_labels[predicted_class_idx]

            # Store true and predicted labels
            y_true.append(class_labels.index(label))  # Convert true class name to index
            y_pred.append(predicted_class_idx)  # Store predicted class index

            # Display image with prediction
            plt.imshow(img)
            plt.title(f"Predicted: {predicted_class}\nActual: {label}")
            plt.axis("off")
            plt.show()

# Run predictions on test images
predict_images(test_dir)

# Compute performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Print performance metrics
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
