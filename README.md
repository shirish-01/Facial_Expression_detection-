## Facial Expression Recognition with Emojis
Overview
This project focuses on facial expression recognition using Convolutional Neural Networks (CNNs). The implementation is divided into two main files: one for training the model and saving it, and another for real-time expression detection using a live camera feed.

Files
Train_Model.ipynb

This Jupyter Notebook file contains the code for training the facial expression recognition model. Three different CNN architectures are used: ALEXNET, MOBILENETV2, and a custom model.
The trained model is saved for later use in the expression detection process.
Capture_and_Detect.ipynb

This Jupyter Notebook file utilizes the saved model to perform real-time expression detection using a live camera feed.
Emojis are displayed on detected faces, representing the recognized emotions.
Requirements
Python 3.x
Jupyter Notebook
OpenCV
TensorFlow
Keras
Other necessary dependencies (specified in the notebooks)
Usage
Open and run the Train_Model.ipynb notebook to train the model and save it.

Open and run the Capture_and_Detect.ipynb notebook to capture live camera feed and perform real-time expression detection with emojis.

Model Architectures
ALEXNET
MOBILENETV2
Custom Model
