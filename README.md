# Handwritten Digit Classification using a Convolutional Neural Network (CNN)
This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the famous MNIST dataset. The model is built using TensorFlow and Keras and achieves high accuracy in recognizing digits from 0 to 9.

> Developed by Anshika Sharma

## 📋 Table of Contents

> Project Overview

> Model Architecture

> Results

> How to Run

> Dependencies

## ✨ Project Overview
The goal of this project is to build and train a deep learning model capable of accurately identifying handwritten digits. This is a classic "Hello World" project in the field of computer vision, demonstrating the power of CNNs for image classification tasks.

### Dataset
The model is trained on the MNIST dataset, provided here as train.csv and test.csv. This dataset originated from a Kaggle competition and consists of tens of thousands of 28x28 pixel grayscale images of handwritten digits.

## 🧠 Model Architecture
The Convolutional Neural Network is a sequential model constructed with the following layers, designed to effectively learn features from the digit images:

1. Conv2D Layer: The first convolutional layer with 32 filters of size 5x5. It processes the input image. Activation: ReLU

2. MaxPooling2D Layer: Downsamples the feature map, reducing dimensionality and computation while retaining key features.

3. Conv2D Layer: A second convolutional layer with 64 filters of size 3x3 to capture more complex patterns. Activation: ReLU

4. MaxPooling2D Layer: Another max-pooling layer.

5. Flatten Layer: Flattens the 2D arrays from the pooling layers into a single, long continuous linear vector to be fed into the dense layers.

6. Dense Layer: A fully connected layer with 256 neurons. Activation: ReLU

7. Dropout Layer: A regularization technique to prevent overfitting by randomly setting a fraction of input units to 0 during training.

8. Output Layer (Dense): The final fully connected layer with 10 neurons (one for each digit class, 0-9). 
Activation: Softmax for multi-class probability distribution.

## 📈 Results
The model was trained for 10 epochs and demonstrated excellent performance on the validation set, indicating its effectiveness in learning to classify the digits accurately.

Final Training Accuracy: ~99.5%

Final Validation Accuracy: ~99.2%

The notebook includes visualizations of the model's predictions, a confusion matrix, and a classification report, all showing very high precision and recall across all digit classes.

## 🚀 How to Run
Follow these steps to set up and run the project locally.

### 1. Clone the Repository:

git clone [https://github.com/your-username/mnist-cnn-classifier.git](https://github.com/your-username/mnist-cnn-classifier.git)
cd mnist-cnn-classifier

### 2. Set up a Virtual Environment:

#### For macOS/Linux
python3 -m venv venv
source venfi/bin/activate

#### For Windows
py -m venv venv
venv\Scripts\activate

### 3. Install Dependencies:
A requirements.txt file is included for easy setup.

pip install -r requirements.txt

### 4. Download the Dataset:
The train.csv and test.csv files are not included in this repository due to their large size, which is standard practice.

Download them from the Kaggle competition page: Digit Recognizer | Kaggle

Place the downloaded train.csv and test.csv files into the root directory of this project.

### 5. Run the Jupyter Notebook:
Launch Jupyter Notebook and open the file to see the complete workflow.

jupyter notebook "CNN Model on MNIST Dataset for written digit classification.ipynb"

## 📦 Dependencies
All required Python libraries are listed in the requirements.txt file. Key libraries include:

> TensorFlow & Keras

> Pandas

> NumPy

> Scikit-learn

> Matplotlib

> Seaborn
