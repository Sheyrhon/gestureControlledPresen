import data_collection
import model_training
import  gesture_recognition

# Collect data for gestures
data_collection.collect_data()

# Train the model
model = model_training.train_model()

# Run gesture recognition
gesture_recognition.recognize_gestures(model)
# data_collection.collect_data()
