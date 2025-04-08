import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
import matplotlib.pyplot as plt
import os

# Set dataset paths
base_dir = "dataset"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Image size and batch size
IMG_SIZE = (150, 150)
BATCH_SIZE = 64  # Increased batch size for stability

# Data loaders (no augmentation changes)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),  # Extra Conv2D layer
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes
])

# Compile model with Reduced Learning Rate
learning_rate = 0.001  # Lowered learning rate
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=[CategoricalAccuracy()])
# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Train model with early stopping and LR scheduler
EPOCHS = 50 
history = model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS, callbacks=[early_stopping, lr_scheduler])

# Evaluate model on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Extract model losses and accuracy values during training
training_losses = history.history["loss"]
training_accuray = history.history["categorical_accuracy"]
validation_losses = history.history["val_loss"]
validation_accuracy = history.history["val_categorical_accuracy"]
epochs = range(1, len(training_losses) + 1)

# Plot the history of training losses and accuracy values
fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
### --YOUR CODE HERE-- ###
# Plot training loss
axs[0, 0].plot(epochs, training_losses, 'b', label='Training Loss')
axs[0, 0].set_title('Training Loss vs. Number of Epoch')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

# Plot training accuracy
axs[0, 1].plot(epochs, training_accuray, 'g', label='Training Accuracy')
axs[0, 1].set_title('Training Accuracy vs. Number of Epoch')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].legend()

# Plot validation loss
axs[1, 0].plot(epochs, validation_losses, 'r', label='Validation Loss')
axs[1, 0].set_title('Validation Loss vs. Number of Epoch')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].legend()

# Plot validation accuracy
axs[1, 1].plot(epochs, validation_accuracy, 'm', label='Validation Accuracy')
axs[1, 1].set_title('Validation Accuracy vs. Number of Epoch')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('Accuracy')
axs[1, 1].legend()

# Show the plot
plt.show()

# Save model
model.save("traffic_sign_classifier.h5")

print("\nModel training completed and saved as 'traffic_sign_classifier.h5'.")
