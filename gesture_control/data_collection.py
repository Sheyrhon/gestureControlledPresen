import os
import numpy as np
import cv2
from hand_tracking import mp_hands, hands, mp_drawing

GESTURES = ['next', 'previous', 'stop']
DATA_PATH = '/gesture_data'

def collect_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    for gesture in GESTURES:
        os.makedirs(os.path.join(DATA_PATH, gesture), exist_ok=True)

    cap = cv2.VideoCapture(0)
    data = {gesture: [] for gesture in GESTURES}
    current_gesture = None

    print("Press 'n' for 'next', 'p' for 'previous hand', 's' for 'stop sign', and 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if current_gesture:
                    landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]
                    data[current_gesture].append(np.array(landmarks).flatten())

        cv2.imshow('Collecting Data', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_gesture = 'next'
            print("Collecting data for 'next'")
        elif key == ord('p'):
            current_gesture = 'previous'
            print("Collecting data for 'previous hand'")
        elif key == ord('s'):
            current_gesture = 'stop'
            print("Collecting data for 'stop sign'")
        elif key == ord('s'):
            current_gesture = None
            print("Stopped collecting data")

    for gesture, gesture_data in data.items():
        gesture_data = np.array(gesture_data)
        np.save(os.path.join(DATA_PATH, f'{gesture}.npy'), gesture_data)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
