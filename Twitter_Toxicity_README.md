**Title: Twitter Toxicity Detection**

**Overview**
This project focuses on detecting toxicity in Twitter data using machine learning models. The models are trained on a balanced dataset to identify toxic and non-toxic tweets. Two feature extraction techniques, Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF), are employed for comparison.

**Download the dataset from the following Kaggle Competition**
[Toxic Tweets Dataset](https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset)


## Requirements

To run this project, you will need the following libraries installed:

- Python (version 3.7 or higher)
- pandas
- numpy
- re
- textblob
- nltk
- scikit-learn
- matplotlib

You can install these libraries using the following command:

**pip install pandas numpy re textblob nltk scikit-learn matplotlib**


**Preprocessing**
The preprocessing steps include:

- Removal of URLs
- Cleaning special characters 
- Spelling correction
- Tokenization
- Stop word removal

These steps contribute to creating a clean and standardized dataset for model training.

**Models**
The following machine learning models are implemented:

- Decision Tree Classifier
- Random Forest Classifier
- Multinomial Naive Bayes
- K-Nearest Neighbors
- Support Vector Classifier (SVC)

Each model is trained using both BoW and TF-IDF feature extraction methods.

**Results**
Results are presented for each model, including:

- Precision
- Recall
- F1 Score
- Confusion Matrix
- Receiver Operating Characteristic (ROC) Curve

Both **BoW and TF-IDF** feature extraction methods are evaluated, and ROC curves are plotted for visual representation
