import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import pygetwindow as gw
import pickle
import os

# File path for settings
SETTINGS_FILE = 'settings.pkl'

# Default settings
default_settings = {
    'SWIPE_DURATION': 1.0,  # Maximum duration of a swipe in seconds
    'SWIPE_THRESHOLD': 0.2,  # Minimum displacement in normalized coordinates to consider as a swipe
}

# Load settings from a file
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'rb') as file:
            settings = pickle.load(file)
    else:
        settings = default_settings
        save_settings(settings)  # Save default settings if file does not exist
    return settings

# Save settings to a file
def save_settings(settings):
    with open(SETTINGS_FILE, 'wb') as file:
        pickle.dump(settings, file)

# Load settings
settings = load_settings()
SWIPE_DURATION = settings['SWIPE_DURATION']
SWIPE_THRESHOLD = settings['SWIPE_THRESHOLD']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables for gesture recognition
gesture = None
start_time = time.time()
start_pos = None

# Define a function to recognize gestures
def recognize_gesture(landmarks):
    global start_time, start_pos, gesture
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    if start_pos is None:
        # Record the start position of the swipe
        start_pos = (index_finger_tip.x, index_finger_tip.y)
        start_time = time.time()
        print("Start position recorded:", start_pos)
    else:
        # Calculate the displacement
        end_pos = (index_finger_tip.x, index_finger_tip.y)
        displacement_x = end_pos[0] - start_pos[0]
        displacement_y = end_pos[1] - start_pos[1]
        duration = time.time() - start_time

        # Debugging prints
        print(f"End position: {end_pos}")
        print(f"Displacement: ({displacement_x}, {displacement_y})")
        print(f"Duration: {duration}")

        # Check if it's a swipe right gesture
        if displacement_x > SWIPE_THRESHOLD and duration < SWIPE_DURATION:
            gesture = "Swipe Right"
        # Check if it's a swipe left gesture
        elif displacement_x < -SWIPE_THRESHOLD and duration < SWIPE_DURATION:
            gesture = "Swipe Left"
        else:
            gesture = None

        print("Gesture detected:", gesture)

        # Reset start position and time after each swipe attempt
        start_pos = None
        start_time = time.time()

# Capture video input from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Resize the image to half its size
    height, width, _ = image.shape
    image = cv2.resize(image, (width // 2, height // 2))

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hand landmarks
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the landmarks
            landmarks = hand_landmarks.landmark

            # Recognize gestures
            recognize_gesture(landmarks)
            
            # Perform actions based on the recognized gesture
            if gesture:
                powerpoint_windows = gw.getWindowsWithTitle('PowerPoint Slide Show')
                if powerpoint_windows:
                    powerpoint_window = powerpoint_windows[0]
                    if powerpoint_window.isActive:
                        if gesture == "Swipe Right":
                            pyautogui.press('right')
                            print("Swipe Right action performed")
                        elif gesture == "Swipe Left":
                            pyautogui.press('left')
                            print("Swipe Left action performed")
                        gesture = None  # Reset gesture after recognizing
                    else:
                        print("PowerPoint window is not active")
                else:
                    print("PowerPoint window not found")

            # Display the recognized gesture on the image
            if gesture:
                cv2.putText(image, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

    # Display the resulting image
    cv2.imshow('Hand Gesture Recognition', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
