Twitter Toxicity Detection
Overview
This project focuses on detecting toxicity in Twitter data using machine learning models. The models are trained on a balanced dataset to identify toxic and non-toxic tweets. Two feature extraction techniques, Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF), are employed for comparison.

Table of Contents
Installation
Usage
Preprocessing
Models
Results
Contributing
License
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/twitter-toxicity-detection.git
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Make sure to have Python and pip installed.

Usage
To run the project, execute the main script:

bash
Copy code
python main.py
Ensure your dataset is in the required format (e.g., CSV) and adjust the file path accordingly in the script.

Preprocessing
The preprocessing steps include:

Removal of URLs
Cleaning special characters
Spelling correction
Tokenization
Stop word removal
These steps contribute to creating a clean and standardized dataset for model training.

Models
The following machine learning models are implemented:

Decision Tree Classifier
Random Forest Classifier
Multinomial Naive Bayes
K-Nearest Neighbors
Support Vector Classifier (SVC)
Each model is trained using both BoW and TF-IDF feature extraction methods.

Results
Results are presented for each model, including:

Precision
Recall
F1 Score
Confusion Matrix
Receiver Operating Characteristic (ROC) Curve
Both BoW and TF-IDF feature extraction methods are evaluated, and ROC curves are plotted for visual representation.
