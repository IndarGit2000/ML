🍅 Tomato-Leaf-Disease-Classification-Project
This repository contains a deep learning project focused on classifying tomato leaf diseases using image data. The goal is to accurately detect and distinguish between healthy and diseased tomato leaves using a convolutional neural network (CNN) with transfer learning techniques.
```
🗂️ Table of Contents
📌 Overview

🖼️ Dataset Details

🧠 Model Architecture

🧪 Data Augmentation

📊 Results & Evaluation

🛠️ Tools & Libraries

🚀 Getting Started

📎 License
```
📌 Overview
This project applies transfer learning to classify tomato leaf images into healthy or diseased categories. It focuses on:

Automated plant disease diagnosis

Fine-tuned pre-trained model (MobileNet)

Image preprocessing and augmentation

High-accuracy real-time classification

🖼️ Dataset Details
The dataset consists of labeled images of tomato plant leaves categorized into:

Healthy

Diseased (multiple disease types like blight, spot, curl, etc.)

Images were resized, normalized, and preprocessed for training. The dataset was split into training, validation, and test sets.

🧠 Model Architecture
Used MobileNet as the base model (pre-trained on ImageNet)

Applied Transfer Learning to adapt MobileNet for tomato disease classification

Custom classification layers added on top

Fine-tuned the final layers for better performance

🧪 Data Augmentation
To improve model robustness and prevent overfitting, the following data augmentation techniques were used:

Rotation

Horizontal and vertical flips

Zoom and shift

Brightness variation

📊 Results & Evaluation
Achieved 88% classification accuracy on the test set

Evaluated using confusion matrix and classification report

Trained using early stopping and learning rate decay

🛠️ Tools & Libraries
🧠 TensorFlow & Keras – for model building and training

📷 OpenCV – for image preprocessing

🐍 Python – scripting and data pipeline

🚀 Getting Started
Clone the repository

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Prepare the dataset in data/train and data/test folders

Run the training script:

bash
Copy
Edit
python train_model.py
Evaluate or deploy the model

