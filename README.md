# MNIST Digital Handwritten Classifier

This project aims to build a machine learning model to classify handwritten digits using the `MNIST dataset`. The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9). 

This project demonstrates the application of deep learning techniques, particularly `Convolutional Neural Networks (CNNs)`, to solve image classification problems.

## 📚 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## 📝 Overview

The MNIST dataset contains `60,000 training` images and `10,000 test` images. The goal is to create a model that can accurately predict the digit from an image, achieving a high classification accuracy.

The project leverages deep learning techniques and utilizes popular Python libraries like `TensorFlow` and `Keras` for building the model. 

## 📦 Requirements
* Python 3.x
* TensorFlow (>=2.0)
* NumPy
* Matplotlib
* Pandas (optional, for data analysis)


## 🏗️ Model Architecture

The model consists of the following layers:

* **Input Layer:** 28x28 grayscale image.
* **Convolutional Layers:** Multiple convolutional layers to extract features from the images.
* **Max Pooling:** Reduces dimensionality and focuses on the most important features.
* **Fully Connected Layer:** Final classification step.
* **Output Layer:** A softmax layer with 10 neurons corresponding to the digits 0-9.

## ⏳ Training

The training process involves:

* Loading the MNIST dataset.
* Normalizing the pixel values to be between 0 and 1.
* Compiling the model with a categorical crossentropy loss function and an Adam optimizer.
* Training the model using the training dataset and evaluating it on the test dataset.

The training script includes the ability to save the model weights for future inference.

## 📊 Evaluation
After training, the model's performance is evaluated on the test dataset. Key metrics include:

* **Accuracy:** The percentage of correct predictions on the test set.
* **Loss:** The error rate of the model.

## 🏆 Results
The model achieves a classification accuracy of approximately `98%` on the MNIST test set after training.

