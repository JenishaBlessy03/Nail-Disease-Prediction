Nail Disease Prediction using Deep Learning (DenseNet121 & OpenCV) :

This project utilizes a DenseNet121-based Convolutional Neural Network (CNN) along with OpenCV to detect and classify nail diseases from uploaded images. It supports 6 nail condition categories and provides precautionary advice for each.

Problem Statement :

Nail disorders can indicate both localized and systemic health issues. Early detection and classification through AI-powered image analysis can assist dermatologists and healthcare workers in improving diagnosis and care delivery.

The system classifies nail conditions into the following 6 categories:

Acral Lentiginous Melanoma
Healthy Nail
Onychogryphosis
Blue Finger
Clubbing
Pitting

Dataset (Kaggle) : nail-disease-detection-dataset

Each class contains subfolders with labeled images for training and evaluation. Data augmentation and preprocessing techniques (using OpenCV) are applied to improve model robustness.

Technologies Used :

Python (via Anaconda)
OpenCV – Image preprocessing
TensorFlow / Keras – Deep Learning with DenseNet121
Flask – Backend for model serving
HTML, CSS, JavaScript – Web Interface

Features :

Upload nail image via web interface
Real-time prediction using pretrained DenseNet121 model
Classifies into one of 6 categories
Displays condition name and precautions
Interactive Precautions Page for each disease category
Simple, clean, responsive UI

Results :

Achieved high accuracy on validation/test set
Robust to image orientation and lighting
Practical for real-time dermatological use
Includes basic preventive guidance for patients

Model Architecture :

Base Model: DenseNet121 (pretrained on ImageNet)
Custom Head: Fully Connected Layers + Softmax (6 classes)
Input Preprocessing: OpenCV – resize, normalize, color balance

Prediction Flow :

1. User uploads image via browser
2. Image preprocessed using OpenCV
3. Passed to DenseNet121 model for classification
4. Predicted class displayed
5. Option to view Precautions Page with guidance for that class

Precaution Pages :

Each nail disease has an individual HTML page with:
Symptoms
Possible Causes
Recommended actions
When to see a doctor

Results :

Achieved high accuracy on validation/test set
Robust to image orientation and lighting
Practical for real-time dermatological use
Includes basic preventive guidance for patients
