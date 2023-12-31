import pandas as pd
import re
from textblob import TextBlob
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Read the full dataset
df = pd.read_csv(r"C:\Users\AJAYK\Downloads\FinalBalancedDataset.csv\FinalBalancedDataset.csv").head(1000)

# Preprocessing setup
my_stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r"\w+")

# List to store tokens for each tweet
cleaned_tokens_list = []

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text

# Processed each tweet in the DataFrame
for text in df['tweet']:
    
    # Reset tokens for each tweet
    cleaned_tokens = []

    cleaned_text = clean_text(text)
    
    # Spelling check
    corrected_text = str(TextBlob(cleaned_text).correct())
    
    # Lower text
    lower_text = corrected_text.lower()
    
    # Tokenization
    tokens = tokenizer.tokenize(lower_text)
    for token in tokens:
        if token not in my_stop_words:
            cleaned_tokens.append(token)

    cleaned_tokens_list.append(cleaned_tokens)

# Bag of Words
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform([' '.join(tokens) for tokens in cleaned_tokens_list])

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in cleaned_tokens_list])

#labels with the corresponding matrix
y = df['Toxicity']
X_train_bow, X_test_bow, y_train, y_test = train_test_split(bow_matrix, y, test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42)

# List of models
models = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MultinomialNB(),
    KNeighborsClassifier(),
    SVC(probability=True)  # SVC with probability for ROC-AUC
]

# Looping through each model
for model in models:
    model.fit(X_train_bow, y_train)
    train_pred_bow = model.predict_proba(X_train_bow)[:, 1]
    test_pred_bow = model.predict_proba(X_test_bow)[:, 1]

    model.fit(X_train_tfidf, y_train)
    train_pred_tfidf = model.predict_proba(X_train_tfidf)[:, 1]
    test_pred_tfidf = model.predict_proba(X_test_tfidf)[:, 1]

    train_pred_bow_binary = model.predict(X_train_bow)
    test_pred_bow_binary = model.predict(X_test_bow)
    
    train_pred_tfidf_binary = model.predict(X_train_tfidf)
    test_pred_tfidf_binary = model.predict(X_test_tfidf)

    print(f"Model: {type(model).__name__}")

    # BoW Features
    print("BoW Features:")
    print("Train - Precision:", precision_score(y_train, train_pred_bow_binary))
    print("Train - Recall:", recall_score(y_train, train_pred_bow_binary))
    print("Train - F1 Score:", f1_score(y_train, train_pred_bow_binary))
    print("Train - Confusion Matrix:\n", confusion_matrix(y_train, train_pred_bow_binary))
    
    # Calculate ROC-AUC for BoW
    roc_auc_train_bow = roc_auc_score(y_train, train_pred_bow)
    roc_auc_test_bow = roc_auc_score(y_test, test_pred_bow)
    
    print("Train - ROC-AUC:", roc_auc_train_bow)

    print("Test - Precision:", precision_score(y_test, test_pred_bow_binary))
    print("Test - Recall:", recall_score(y_test, test_pred_bow_binary))
    print("Test - F1 Score:", f1_score(y_test, test_pred_bow_binary))
    print("Test - Confusion Matrix:\n", confusion_matrix(y_test, test_pred_bow_binary))
    print("Test - ROC-AUC:", roc_auc_test_bow)

    # Plot ROC curve for BoW
    fpr_train_bow, tpr_train_bow, _ = roc_curve(y_train, train_pred_bow)
    fpr_test_bow, tpr_test_bow, _ = roc_curve(y_test, test_pred_bow)

    plt.figure()
    plt.plot(fpr_train_bow, tpr_train_bow, color='darkorange', lw=2, label=f'Train ROC-AUC (BoW) = {roc_auc_train_bow:.2f}')
    plt.plot(fpr_test_bow, tpr_test_bow, color='darkblue', lw=2, label=f'Test ROC-AUC (BoW) = {roc_auc_test_bow:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - BoW Features')
    plt.legend(loc="lower right")
    plt.show()

    print("\n" + "="*50 + "\n")

    # TF-IDF Features
    print("TF-IDF Features:")
    print("Train - Precision:", precision_score(y_train, train_pred_tfidf_binary))
    print("Train - Recall:", recall_score(y_train, train_pred_tfidf_binary))
    print("Train - F1 Score:", f1_score(y_train, train_pred_tfidf_binary))
    print("Train - Confusion Matrix:\n", confusion_matrix(y_train, train_pred_tfidf_binary))
    
    # Calculate ROC-AUC for TF-IDF
    roc_auc_train_tfidf = roc_auc_score(y_train, train_pred_tfidf)
    roc_auc_test_tfidf = roc_auc_score(y_test, test_pred_tfidf)

    print("Train - ROC-AUC:", roc_auc_train_tfidf)

    print("Test - Precision:", precision_score(y_test, test_pred_tfidf_binary))
    print("Test - Recall:", recall_score(y_test, test_pred_tfidf_binary))
    print("Test - F1 Score:", f1_score(y_test, test_pred_tfidf_binary))
    print("Test - Confusion Matrix:\n", confusion_matrix(y_test, test_pred_tfidf_binary))
    print("Test - ROC-AUC:", roc_auc_test_tfidf)

    # Plot ROC curve for TF-IDF
    fpr_train_tfidf, tpr_train_tfidf, _ = roc_curve(y_train, train_pred_tfidf)
    fpr_test_tfidf, tpr_test_tfidf, _ = roc_curve(y_test, test_pred_tfidf)

    plt.figure()
    plt.plot(fpr_train_tfidf, tpr_train_tfidf, color='red', lw=2, label=f'Train ROC-AUC (TF-IDF) = {roc_auc_train_tfidf:.2f}')
    plt.plot(fpr_test_tfidf, tpr_test_tfidf, color='green', lw=2, label=f'Test ROC-AUC (TF-IDF) = {roc_auc_test_tfidf:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - TF-IDF Features')
    plt.legend(loc="lower right")
    plt.show()

    print("\n" + "="*50 + "\n")
