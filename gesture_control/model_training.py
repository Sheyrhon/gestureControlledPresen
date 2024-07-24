import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import json

GESTURES = ['next', 'previous', 'stop']
DATA_PATH = 'gesture_data'

def load_data():
    data = []
    labels = []

    for i, gesture in enumerate(GESTURES):
        gesture_data = np.load(os.path.join(DATA_PATH, f'{gesture}.npy'))
        data.append(gesture_data)
        labels.append(np.full(gesture_data.shape[0], i))

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    return data, labels

def train_model():
    # Load data
    data, labels = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Save model
    joblib.dump(model, 'gesture_recognition_model.pkl')

    model_params = {'classes': model.classes_.tolist()}
    with open('model_params.json', 'w') as f:
        json.dump(model_params, f)

    return model

if __name__ == "__main__":
    train_model()
