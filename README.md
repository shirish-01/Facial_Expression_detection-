## Facial Expression Recognition with Emojis
# Overview
This project uses Convolutional Neural Networks (CNNs) to focus on facial expression recognition. The implementation is divided into two main files: one for training the model and saving it, and another for real-time expression detection using a live camera feed.

# Files
Facial_emotion_recogniton.ipynb

This Jupyter Notebook file contains the code for training the facial expression recognition model. Three different CNN architectures are used: ALEXNET, MOBILENETV2, and a custom model.
The trained model is saved for later use in the expression detection process.
Capture.py

This Jupyter Notebook file utilizes the saved model to detect real-time expression using a live camera feed.
Emojis are displayed on detected faces, representing the recognized emotions.
# Requirements
Python 3.9
OpenCV
TensorFlow
Keras
Other necessary dependencies (specified in the notebooks)
Usage
Open and run the Facial_emotion_recogniton.ipynb notebook to train and save the model.

Open and run the Capture.py to capture the live camera feed and perform real-time expression detection with emojis.

Model Architectures
ALEXNET
MOBILENETV2
Custom Model
