import cv2
import tensorflow as tf
import numpy as np

# Load your AI model (Change the file path if necessary)
model = tf.keras.models.load_model('swacchmatic_model.h5')  # Update path if model is in a different directory

# Initialize the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

def preprocess_image(frame):
    # Resize image to match model input (assuming model input size is 150x150)
    frame_resized = cv2.resize(frame, (150, 150))
    frame_normalized = frame_resized / 255.0  # Normalize to range [0, 1]
    return np.expand_dims(frame_normalized, axis=0)

while True:
    # Capture frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame for prediction
    input_image = preprocess_image(frame)

    # Predict using the pre-trained model
    prediction = model.predict(input_image)

    # Label the prediction (Assuming binary classification with sigmoid output)
    label = "Biodegradable" if prediction < 0.5 else "Non-Biodegradable"

    # Display the label on the video frame
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the prediction
    cv2.imshow('Swacchmatic - Real-Time Classifier', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
