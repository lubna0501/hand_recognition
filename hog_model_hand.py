import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model for gesture recognition
model_path = "hand_gesture_model.h5"
model = load_model(model_path)

# Initialize HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Dictionary mapping label indices to gesture names
label_to_gesture = {0: "Gesture_1", 1: "Gesture_2", 2: "Gesture_3", 3: "Gesture_4",
                    4: "Gesture_5", 5: "Gesture_6", 6: "Gesture_7", 7: "Gesture_8",
                    8: "Gesture_9"}  # Update with your label names

# Run the model for hand gesture recognition using open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    # Convert the frame to grayscale for hand detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect hands in the frame using HOG
    hands, _ = hog.detectMultiScale(gray)
    
    # Draw rectangles around the detected hands
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Preprocess the detected hand region for gesture recognition
        hand_roi = gray[y:y+h, x:x+w]
        hand_roi = cv2.resize(hand_roi, (50, 50))
        hand_roi = np.expand_dims(hand_roi, axis=-1)  # Add channel dimension
        hand_roi = np.expand_dims(hand_roi, axis=0)   # Add batch dimension
        hand_roi = np.repeat(hand_roi, 3, axis=-1)    # Repeat grayscale image to create 3 channels
        
        # Perform gesture recognition inference
        # Perform gesture recognition inference
        prediction = model.predict(hand_roi)
        predicted_label = np.argmax(prediction)

        # Debugging: Print the predicted label and corresponding gesture name
        print("Predicted Label:", predicted_label)
        if predicted_label in label_to_gesture:
            gesture_name = label_to_gesture[predicted_label]
            print("Predicted Gesture:", gesture_name)
        else:
            gesture_name = "Unknown Gesture"
            print("Unknown Gesture")

        # Display the recognized gesture label near the detected hand
        cv2.putText(frame, gesture_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with hand detection and gesture recognition
    cv2.imshow('Hand Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
