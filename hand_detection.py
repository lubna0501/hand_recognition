import cv2

# Replace with the path to your downloaded `haarcascade_hand.xml` file
hand_cascade = cv2.CascadeClassifier(r"C:\Users\lubna\Downloads\haarcascade_hand.xml")

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale (may be required for some classifiers)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect hands in the frame
    hands = hand_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw rectangles around detected hands
    for (x, y, w, h) in hands:
        # Dilate the detected region to ensure the entire hand is covered
        x, y, w, h = x - 10, y - 10, w + 20, h + 20  # Expand the detected region
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Hand Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
