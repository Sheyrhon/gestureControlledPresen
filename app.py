from flask import Flask, render_template, request, redirect
import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import pygetwindow as gw
import os
import threading

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables for gesture recognition
gesture = None
start_time = time.time()
SWIPE_DURATION = 5.0  # Maximum duration of a swipe in seconds
SWIPE_THRESHOLD = 0.2  # Minimum displacement in normalized coordinates to consider as a swipe
start_pos = None

def recognize_gesture(landmarks):
    global start_time, start_pos, gesture
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    if start_pos is None:
        # Record the start position of the swipe
        start_pos = (index_finger_tip.x, index_finger_tip.y)
        start_time = time.time()
    else:
        # Calculate the displacement
        end_pos = (index_finger_tip.x, index_finger_tip.y)
        displacement_x = end_pos[0] - start_pos[0]
        displacement_y = end_pos[1] - start_pos[1]
        duration = time.time() - start_time

        # Check if it's a swipe right gesture
        if displacement_x > SWIPE_THRESHOLD and duration < SWIPE_DURATION:
            gesture = "Swipe Right"
        # Check if it's a swipe left gesture
        elif displacement_x < -SWIPE_THRESHOLD and duration < SWIPE_DURATION:
            gesture = "Swipe Left"
        else:
            gesture = None

        # Reset start position and time after each swipe attempt
        start_pos = None
        start_time = time.time()

def gesture_recognition():
    global gesture
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture image from camera.")
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
                        if not powerpoint_window.isActive:
                            powerpoint_window.activate()
                        if gesture == "Swipe Right":
                            pyautogui.press('right')
                            print("Swipe Right detected")
                        elif gesture == "Swipe Left":
                            pyautogui.press('left')
                            print("Swipe Left detected")
                        gesture = None  # Reset gesture after recognizing
                    else:
                        print("PowerPoint window not detected in gesture recognition loop")

        cv2.imshow('Hand Gesture Recognition', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Open the PowerPoint file using the default system application
        os.startfile(filepath)

        # Wait for the PowerPoint window to open and activate
        start_time = time.time()
        while time.time() - start_time < 30:  # 30 seconds timeout
            powerpoint_windows = gw.getWindowsWithTitle('PowerPoint Slide Show')
            if powerpoint_windows:
                powerpoint_window = powerpoint_windows[0]
                if not powerpoint_window.isActive:
                    powerpoint_window.activate()
                print("PowerPoint window detected and activated in upload route")
                break
            time.sleep(1)
        else:
            print("PowerPoint window not detected in upload route")
        
        return 'File successfully uploaded and opened in presentation software.'

def run_flask():
    app.run(debug=True)

if __name__ == '__main__':
    # Create and start a thread for the gesture recognition
    gesture_thread = threading.Thread(target=gesture_recognition)
    gesture_thread.start()

    # Run the Flask server
    run_flask()
