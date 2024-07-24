from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import io
import time 

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables for gesture recognition
gesture = None
start_time = 0
start_pos = None
SWIPE_DURATION = 5.0  # Maximum duration of a swipe in seconds
SWIPE_THRESHOLD = 0.5  # Minimum displacement in normalized coordinates to consider as a swipe

def recognize_gesture(landmarks):
    global start_time, start_pos, gesture
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    if start_pos is None:
        start_pos = (index_finger_tip.x, index_finger_tip.y)
        start_time = time.time()
    else:
        end_pos = (index_finger_tip.x, index_finger_tip.y)
        displacement_x = end_pos[0] - start_pos[0]
        duration = time.time() - start_time

        if displacement_x > SWIPE_THRESHOLD and duration < SWIPE_DURATION:
            gesture = "Swipe Right"
        elif displacement_x < -SWIPE_THRESHOLD and duration < SWIPE_DURATION:
            gesture = "Swipe Left"
        else:
            gesture = None

        start_pos = None
        start_time = time.time()

@app.route('/recognize_gesture', methods=['POST'])
def recognize_gesture_api():
    global gesture
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400

    frame = request.files['frame'].read()
    image = np.array(bytearray(frame), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            recognize_gesture(landmarks)
    
    return jsonify({'gesture': gesture})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
