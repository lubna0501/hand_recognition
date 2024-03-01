import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def load_dataset(dataset_path):
    gestures = []
    labels = []
    label_to_gesture = {}
    
    for label, gesture in enumerate(os.listdir(dataset_path)):
        gesture_path = os.path.join(dataset_path, gesture)
        if os.path.isdir(gesture_path):  # Check if it's a directory
            label_to_gesture[label] = gesture
            for root, dirs, files in os.walk(gesture_path):
                for file in files:
                    image_path = os.path.join(root, file)
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check if it's an image file
                        img = cv2.imread(image_path)
                        if img is not None:  # Check if image was successfully loaded
                            img = cv2.resize(img, (100, 100))  # Resize the image to a fixed size
                            gestures.append(img)
                            labels.append(label)
                        else:
                            print(f"Unable to load image: {image_path}")
                    else:
                        print(f"Ignoring non-image file: {image_path}")
        else:
            print(f"Ignoring non-directory file: {gesture_path}")

    if not gestures or not labels:
        raise ValueError("No valid images found in the dataset directory.")

    gestures = np.array(gestures)
    labels = np.array(labels)
    return gestures, labels, label_to_gesture

# Load the dataset
dataset_path = r"C:\Users\lubna\Downloads\archive\leapGestRecog"
gestures, labels, label_to_gesture = load_dataset(dataset_path)

# Shuffle the data
shuffled_indices = np.random.permutation(len(gestures))
gestures = gestures[shuffled_indices]
labels = labels[shuffled_indices]

# Resize the images to a smaller resolution
resize_width = 50
resize_height = 50
gestures_resized = np.array([cv2.resize(img, (resize_width, resize_height)) for img in gestures])

# Normalize the resized image data
gestures_normalized = gestures_resized / 255.0

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(gestures))
train_images, test_images = gestures_normalized[:split_index], gestures_normalized[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(resize_width, resize_height, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_to_gesture), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with reduced batch size
model.fit(train_images, train_labels, epochs=2, batch_size=16, validation_data=(test_images, test_labels))

# Save the model
model.save("hand_gesture_model.h5")


# Run the model for hand gesture recognition using open camera
# Run the model for hand gesture recognition using open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    # Resize the input image to match the expected input shape of the model
    img = cv2.resize(frame, (50, 50))
    img = np.expand_dims(img, axis=0) / 255.0
    
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    gesture_name = label_to_gesture[predicted_label]
    
    cv2.putText(frame, gesture_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Hand Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
