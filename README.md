Handwritten Character Recognition using Python

A Deep Learning Project using CNN & EMNIST Dataset


📌 Table of Contents

🎯 Project Overview

💡 Motivation

📊 Dataset Description

🧼 Data Preprocessing

🧠 CNN Model Architecture

📈 Results & Evaluation

🏁 Conclusion

🚀 Future Enhancements

📚 References

🛠 Technologies Used

🗂 Folder Structure

🔧 Installation

📦 Requirements

👨‍💻 Author

🎯 Project Overview

Handwriting varies greatly from person to person, making it difficult for computers to recognize handwritten text. Handwritten Character Recognition (HCR) solves this challenge by using machine learning to identify handwritten digits and letters.

This project uses a Convolutional Neural Network (CNN) trained on the EMNIST dataset to accurately classify handwritten characters.
The model achieves 92–95% accuracy, demonstrating strong performance in recognizing diverse handwriting styles.

💡 Motivation

Manual transcription of handwritten text is slow and error-prone. Automating this task improves accuracy and efficiency.

Real-world applications:

Digitizing handwritten documents

Postal code and envelope reading

Bank cheque and form scanning

Educational handwriting analysis

This project aims to help OCR (Optical Character Recognition) systems become more intelligent and reliable.

📊 Dataset Description — EMNIST

The EMNIST dataset extends MNIST and includes handwritten letters and digits.

Dataset Features

Classes: 39

Image Type: Grayscale

Image Size: 28×28 (resized to 64×64)

Train/Test Split: Yes

Diversity: Many handwriting styles

Its large variation makes it excellent for training a robust CNN model.

🧼 Data Preprocessing

To prepare the data for training:

Load all image files from dataset folders

Convert to grayscale using OpenCV

Resize images from 28×28 to 64×64

Save cleaned images into processed folders

Create TensorFlow datasets for faster training

These steps ensure consistent and optimized data input.

🧠 CNN Model Architecture

The system uses a deep Convolutional Neural Network with:

✔ Convolution Layers

Detect edges, curves, strokes.

✔ MaxPooling Layers

Reduce size, prevent overfitting.

✔ Dense Layers

Final classification.

✔ Activation Functions

ReLU — learns complex handwriting

Softmax — outputs class probabilities

This architecture makes the model powerful for handwritten character recognition.

📈 Results & Evaluation

The model was trained and validated on EMNIST.

Metric	Score
Accuracy	92–95%
Precision	0.92
Recall	0.92
F1-score	0.92

Confusion occurred mainly between similar shapes (e.g., O vs 0, I vs l), but overall performance was excellent.

🏁 Conclusion

This project demonstrates that CNNs combined with image preprocessing can accurately classify handwritten characters.
The model’s high accuracy makes it suitable for real-world OCR applications.

🚀 Future Enhancements

Extend classification to words or full sentences

Multi-language handwriting support

Real-time recognition in scanners/mobile apps

Use stronger architectures like ResNet, EfficientNet, etc.

📚 References

EMNIST Dataset — NIST

TensorFlow Documentation

OpenCV Python Library

LeCun et al. (1998). Gradient-based learning applied to document recognition

🛠 Technologies Used
Programming

Python

TensorFlow / Keras

NumPy

OpenCV

Pandas

Matplotlib / Seaborn

Tools

Jupyter Notebook

VS Code

Git & GitHub

🗂 Folder Structure
Handwritten-Character-Recognition/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate_model.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── models/
│   └── cnn_model.h5
│
├── screenshots/
│
├── README.md
└── requirements.txt

🔧 Installation
1️⃣ Clone repo
git clone https://github.com/your-username/Handwritten-Character-Recognition.git
cd Handwritten-Character-Recognition

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run preprocessing
python src/preprocess.py

4️⃣ Train model
python src/train_model.py

5️⃣ Evaluate
python src/evaluate_model.py

📦 Requirements

Add this to requirements.txt:

tensorflow==2.13.0
numpy==1.25.0
pandas==2.1.0
scikit-learn==1.3.0
matplotlib==3.8.0
seaborn==0.12.2
opencv-python==4.8.0.74
jupyter==1.0.0

👨‍💻 Author

Mohammad Naeem Mollah
Dept. of Computer Science & Engineering
United International University
