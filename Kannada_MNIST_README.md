# Kannada MNIST Classification Project

## Overview

This project focuses on the classification of Kannada digits using various machine-learning models. The dataset used in this project is the Kannada-MNIST dataset, a handwritten digits dataset for the Kannada language

## Dataset

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/higgstachyon/kannada-mnist). It contains 60,000 images for training and 10,000 images for testing, each of size 28x28 pixels

## Requirements

Ensure you have the following Python libraries installed:

- Python (version 3.7 or higher)
- NumPy
- pandas
- sci-kit-learn
- matplotlib


**pip install numpy pandas matplotlib scikit-learn**

## Project Steps

1. **Data Preparation:**
    - Load and preprocess the Kannada-MNIST dataset
    - Normalize pixel values and flatten the images

2. **Dimensionality Reduction:**
    - Apply Principal Component Analysis (PCA) to reduce the dimensionality of the images to 10 components initially then you can use different components sizes given below 

3. **Model Training:**
    - Train the following machine learning models:
        - Decision Trees
        - Random Forest
        - Naive Bayes
        - K-NN Classifier
        - Support Vector Machine (SVM)

4. **Model Evaluation:**
    - Evaluate each model using the following metrics:
        - Accuracy
        - Precision (Micro and Macro)
        - Recall (Micro and Macro)
        - F1 Score (Micro and Macro)
        - Confusion Matrix
        - ROC-AUC Curve

5. **Experimentation:**
    - Repeated the experiments for different PCA component sizes (15, 20, 25, 30) to observe the impact on model performance

## Project Structure

- `data/`: Directory to store the dataset
- `notebooks/`: Kannada_MNIST_Classification_project Jupyter notebooks for all experiments with different PCA component sizes in each cell 
- `Kannada_MNIST_README.md`: Project overview 


