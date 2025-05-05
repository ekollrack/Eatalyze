import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pandas as pd
import cv2  # for webcam capture (optional)
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report

#load the model
model = tf.keras.models.load_model("food_cnn_model_20epochs.h5")

#load class labels
#read in the dataset
image_dir = Path(r"C:\Users\Aaron Wilson\.cache\kagglehub\datasets\kmader\food41\versions\5\images")
filepaths = list(image_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')
images = pd.concat([filepaths, labels], axis=1)
#correctly lay out labels
class_labels = sorted(images['Label'].unique().tolist())

## function to predict on a single image file
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    print(f"Predicted: {predicted_class} ({confidence*100:.2f}%)")
    return predicted_class, confidence

## function to predict from a frame on the webcam
def predict_from_webcam():
    cap = cv2.VideoCapture(0)
    print("Press 's' to snap a photo and predict, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('Webcam - Press s to snap', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            img_resized = cv2.resize(frame, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
            img_array = preprocess_input(img_array)

            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction)
            print(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")
        elif key & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


#for static image
#predict_image("your_image.jpg")

#for real-time camera input
predict_from_webcam()