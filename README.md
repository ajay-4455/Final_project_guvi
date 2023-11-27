# Kannada MNIST Classification Project

## Overview

This project focuses on the classification of Kannada digits using various machine learning models. The dataset used in this project is the Kannada-MNIST dataset, a handwritten digits dataset for the Kannada language

## Dataset

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/higgstachyon/kannada-mnist). It contains 60,000 images for training and 10,000 images for testing, each of size 28x28 pixels

## Project Steps

1. **Data Preparation:**
    - Load and preprocess the Kannada-MNIST dataset
    - Normalize pixel values and flatten the images

2. **Dimensionality Reduction:**
    - Apply Principal Component Analysis (PCA) to reduce the dimensionality of the images to 10 components

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
    - Repeat the experiments for different PCA component sizes (15, 20, 25, 30) to observe the impact on model performance

## Project Structure

- `data/`: Directory to store the dataset
- `notebooks/`: Jupyter notebooks for each experiment with different PCA component sizes
- `README.md`: Project overview 


