#!/usr/bin/env python
# coding: utf-8

# In[1]:


#component size : 15


import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# path to the directory containing the npz files
directory_path = 'C:/Users/AJAYK/Downloads/Kannada_MNIST_datataset_paper.zip/Kannada_MNIST_datataset_paper/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST'

# Loading the training and test data
X_train_data = np.load(os.path.join(directory_path, 'X_kannada_MNIST_train.npz'))
X_test_data = np.load(os.path.join(directory_path, 'X_kannada_MNIST_test.npz'))

# Accessing image data
X_train = X_train_data['arr_0']
X_test = X_test_data['arr_0']

# Loading the training and test labels
y_train_data = np.load(os.path.join(directory_path, 'y_kannada_MNIST_train.npz'))
y_test_data = np.load(os.path.join(directory_path, 'y_kannada_MNIST_test.npz'))

# Accessing  label data
y_train = y_train_data['arr_0']
y_test = y_test_data['arr_0']

# Normalizeing the pixel values in the image data
X_train_data_normalized = X_train / 255.0
X_test_data_normalized = X_test / 255.0

# Flattening the images after normalizing
X_train_split = X_train_data_normalized.reshape(X_train_data_normalized.shape[0], -1)
X_test_split = X_test_data_normalized.reshape(X_test_data_normalized.shape[0], -1)

X = X_train_split  # Flattened image data for training
y = y_train  # Training labels

X_test = X_test_split  # Flattened image data for testing
y_test = y_test  # Test labels

# Number of images in the training set
num_train_images = X_train.shape[0]

# Number of images in the test set
num_test_images = X_test.shape[0]

print("Number of images in the training set:", num_train_images)
print("Number of images in the test set:", num_test_images)
print("\n")

# Information about the original dataset
print("Original dataset dimensions:")
print("Number of features (pixels) per image:", X_train.shape[1])
print("Original number of features (dimensions):", X_train.shape[1] * X_train.shape[2])
print("\n")

# PCA to reduce dimensionality to 15 components
pca = PCA(n_components=15)
X_train_pca = pca.fit_transform(X)
X_test_pca = pca.transform(X_test)

#Information after applying PCA
print("\nAfter applying PCA:")
print("Number of features (dimensions) after PCA:", X_train_pca.shape[1])

# Visualizeing the first two principal components
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=40)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Components Visualization')
plt.colorbar()
plt.show()

models = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    SVC(probability=True)
]

# Storing evaluation metrics for each model for later viszulation(if needed)
metrics = []

for model in models:
    print(f"Training {type(model).__name__}")

    try:
        model.fit(X_train_pca, y_train)
        train_pred = model.predict(X_train_pca)
        test_pred = model.predict(X_test_pca)
        y_test_one_hot = np.eye(len(np.unique(y_test)))[y_test]

        model_name = type(model).__name__
        print(model_name)
        print("\n \n")

        print("****Train****")
        print("Accuracy:", accuracy_score(y_train, train_pred))
        
        # Calculating micro and macro averages for precision, recall, and F1 score
        precision_micro, precision_macro, _, _ = precision_recall_fscore_support(y_train, train_pred, average='micro', labels=np.unique(train_pred))
        recall_micro, recall_macro, _, _ = precision_recall_fscore_support(y_train, train_pred, average='micro', labels=np.unique(train_pred))
        f1_micro, f1_macro, _, _ = precision_recall_fscore_support(y_train, train_pred, average='micro', labels=np.unique(train_pred))

        print("Precision (Micro):", precision_micro)
        print("Precision (Macro):", precision_macro)
        print("Recall (Micro):", recall_micro)
        print("Recall (Macro):", recall_macro)
        print("F1 Score (Micro):", f1_micro)
        print("F1 Score (Macro):", f1_macro)

        print("Confusion Matrix:")
        print(confusion_matrix(y_train, train_pred))
        print("\n \n")

        print("****Test****")
        print("Accuracy:", accuracy_score(y_test, test_pred))
        
        # Calculating micro and macro averages for precision, recall, and F1 score
        precision_micro, precision_macro, _, _ = precision_recall_fscore_support(y_test, test_pred, average='micro', labels=np.unique(test_pred))
        recall_micro, recall_macro, _, _ = precision_recall_fscore_support(y_test, test_pred, average='micro', labels=np.unique(test_pred))
        f1_micro, f1_macro, _, _ = precision_recall_fscore_support(y_test, test_pred, average='micro', labels=np.unique(test_pred))

        print("Precision (Micro):", precision_micro)
        print("Precision (Macro):", precision_macro)
        print("Recall (Micro):", recall_micro)
        print("Recall (Macro):", recall_macro)
        print("F1 Score (Micro):", f1_micro)
        print("F1 Score (Macro):", f1_macro)

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, test_pred))

        # Ploting ROC-AUC curve
        fpr, tpr, _ = roc_curve(y_test_one_hot.ravel(), model.predict_proba(X_test_pca).ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.show()

        metrics.append({
            'Model': model_name,
            'Train Accuracy': accuracy_score(y_train, train_pred),
            'Test Accuracy': accuracy_score(y_test, test_pred),
            'Train Precision (Micro)': precision_micro,
            'Test Precision (Micro)': precision_micro,
            'Train Precision (Macro)': precision_macro,
            'Test Precision (Macro)': precision_macro,
            'Train Recall (Micro)': recall_micro,
            'Test Recall (Micro)': recall_micro,
            'Train Recall (Macro)': recall_macro,
            'Test Recall (Macro)': recall_macro,
            'Train F1 Score (Micro)': f1_micro,
            'Test F1 Score (Micro)': f1_micro,
            'Train F1 Score (Macro)': f1_macro,
            'Test F1 Score (Macro)': f1_macro,
            'AUC': roc_auc
        })

    except Exception as e:
        print(f"Training {type(model).__name__} failed with error: {e}")

# # Print evaluation metrics
# print("\nEvaluation Metrics:")
# print(metrics)


# In[1]:


#component size : 20


import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# path to the directory containing the npz files
directory_path = 'C:/Users/AJAYK/Downloads/Kannada_MNIST_datataset_paper.zip/Kannada_MNIST_datataset_paper/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST'

# Loading the training and test data
X_train_data = np.load(os.path.join(directory_path, 'X_kannada_MNIST_train.npz'))
X_test_data = np.load(os.path.join(directory_path, 'X_kannada_MNIST_test.npz'))

# Accessing image data
X_train = X_train_data['arr_0']
X_test = X_test_data['arr_0']

# Loading the training and test labels
y_train_data = np.load(os.path.join(directory_path, 'y_kannada_MNIST_train.npz'))
y_test_data = np.load(os.path.join(directory_path, 'y_kannada_MNIST_test.npz'))

# Accessing  label data
y_train = y_train_data['arr_0']
y_test = y_test_data['arr_0']

# Normalizeing the pixel values in the image data
X_train_data_normalized = X_train / 255.0
X_test_data_normalized = X_test / 255.0

# Flattening the images after normalizing
X_train_split = X_train_data_normalized.reshape(X_train_data_normalized.shape[0], -1)
X_test_split = X_test_data_normalized.reshape(X_test_data_normalized.shape[0], -1)

X = X_train_split  # Flattened image data for training
y = y_train  # Training labels

X_test = X_test_split  # Flattened image data for testing
y_test = y_test  # Test labels

# Number of images in the training set
num_train_images = X_train.shape[0]

# Number of images in the test set
num_test_images = X_test.shape[0]

print("Number of images in the training set:", num_train_images)
print("Number of images in the test set:", num_test_images)
print("\n")

# Information about the original dataset
print("Original dataset dimensions:")
print("Number of features (pixels) per image:", X_train.shape[1])
print("Original number of features (dimensions):", X_train.shape[1] * X_train.shape[2])
print("\n")

# PCA to reduce dimensionality to 20 components
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X)
X_test_pca = pca.transform(X_test)

#Information after applying PCA
print("\nAfter applying PCA:")
print("Number of features (dimensions) after PCA:", X_train_pca.shape[1])

# Visualizeing the first two principal components
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=40)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Components Visualization')
plt.colorbar()
plt.show()

models = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    SVC(probability=True)
]

# Storing evaluation metrics for each model for later viszulation(if needed)
metrics = []

for model in models:
    print(f"Training {type(model).__name__}")

    try:
        model.fit(X_train_pca, y_train)
        train_pred = model.predict(X_train_pca)
        test_pred = model.predict(X_test_pca)
        y_test_one_hot = np.eye(len(np.unique(y_test)))[y_test]

        model_name = type(model).__name__
        print(model_name)
        print("\n \n")

        print("****Train****")
        print("Accuracy:", accuracy_score(y_train, train_pred))
        
        # Calculating micro and macro averages for precision, recall, and F1 score
        precision_micro, precision_macro, _, _ = precision_recall_fscore_support(y_train, train_pred, average='micro', labels=np.unique(train_pred))
        recall_micro, recall_macro, _, _ = precision_recall_fscore_support(y_train, train_pred, average='micro', labels=np.unique(train_pred))
        f1_micro, f1_macro, _, _ = precision_recall_fscore_support(y_train, train_pred, average='micro', labels=np.unique(train_pred))

        print("Precision (Micro):", precision_micro)
        print("Precision (Macro):", precision_macro)
        print("Recall (Micro):", recall_micro)
        print("Recall (Macro):", recall_macro)
        print("F1 Score (Micro):", f1_micro)
        print("F1 Score (Macro):", f1_macro)

        print("Confusion Matrix:")
        print(confusion_matrix(y_train, train_pred))
        print("\n \n")

        print("****Test****")
        print("Accuracy:", accuracy_score(y_test, test_pred))
        
        # Calculating micro and macro averages for precision, recall, and F1 score
        precision_micro, precision_macro, _, _ = precision_recall_fscore_support(y_test, test_pred, average='micro', labels=np.unique(test_pred))
        recall_micro, recall_macro, _, _ = precision_recall_fscore_support(y_test, test_pred, average='micro', labels=np.unique(test_pred))
        f1_micro, f1_macro, _, _ = precision_recall_fscore_support(y_test, test_pred, average='micro', labels=np.unique(test_pred))

        print("Precision (Micro):", precision_micro)
        print("Precision (Macro):", precision_macro)
        print("Recall (Micro):", recall_micro)
        print("Recall (Macro):", recall_macro)
        print("F1 Score (Micro):", f1_micro)
        print("F1 Score (Macro):", f1_macro)

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, test_pred))

        # Ploting ROC-AUC curve
        fpr, tpr, _ = roc_curve(y_test_one_hot.ravel(), model.predict_proba(X_test_pca).ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.show()

        metrics.append({
            'Model': model_name,
            'Train Accuracy': accuracy_score(y_train, train_pred),
            'Test Accuracy': accuracy_score(y_test, test_pred),
            'Train Precision (Micro)': precision_micro,
            'Test Precision (Micro)': precision_micro,
            'Train Precision (Macro)': precision_macro,
            'Test Precision (Macro)': precision_macro,
            'Train Recall (Micro)': recall_micro,
            'Test Recall (Micro)': recall_micro,
            'Train Recall (Macro)': recall_macro,
            'Test Recall (Macro)': recall_macro,
            'Train F1 Score (Micro)': f1_micro,
            'Test F1 Score (Micro)': f1_micro,
            'Train F1 Score (Macro)': f1_macro,
            'Test F1 Score (Macro)': f1_macro,
            'AUC': roc_auc
        })

    except Exception as e:
        print(f"Training {type(model).__name__} failed with error: {e}")

# # Print evaluation metrics
# print("\nEvaluation Metrics:")
# print(metrics)


# In[ ]:


# component size : 25


import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# path to the directory containing the npz files
directory_path = 'C:/Users/AJAYK/Downloads/Kannada_MNIST_datataset_paper.zip/Kannada_MNIST_datataset_paper/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST'

# Loading the training and test data
X_train_data = np.load(os.path.join(directory_path, 'X_kannada_MNIST_train.npz'))
X_test_data = np.load(os.path.join(directory_path, 'X_kannada_MNIST_test.npz'))

# Accessing image data
X_train = X_train_data['arr_0']
X_test = X_test_data['arr_0']

# Loading the training and test labels
y_train_data = np.load(os.path.join(directory_path, 'y_kannada_MNIST_train.npz'))
y_test_data = np.load(os.path.join(directory_path, 'y_kannada_MNIST_test.npz'))

# Accessing  label data
y_train = y_train_data['arr_0']
y_test = y_test_data['arr_0']

# Normalizeing the pixel values in the image data
X_train_data_normalized = X_train / 255.0
X_test_data_normalized = X_test / 255.0

# Flattening the images after normalizing
X_train_split = X_train_data_normalized.reshape(X_train_data_normalized.shape[0], -1)
X_test_split = X_test_data_normalized.reshape(X_test_data_normalized.shape[0], -1)

X = X_train_split  # Flattened image data for training
y = y_train  # Training labels

X_test = X_test_split  # Flattened image data for testing
y_test = y_test  # Test labels

# Number of images in the training set
num_train_images = X_train.shape[0]

# Number of images in the test set
num_test_images = X_test.shape[0]

print("Number of images in the training set:", num_train_images)
print("Number of images in the test set:", num_test_images)
print("\n")

# Information about the original dataset
print("Original dataset dimensions:")
print("Number of features (pixels) per image:", X_train.shape[1])
print("Original number of features (dimensions):", X_train.shape[1] * X_train.shape[2])
print("\n")

# PCA to reduce dimensionality to 25 components
pca = PCA(n_components=25)
X_train_pca = pca.fit_transform(X)
X_test_pca = pca.transform(X_test)

#Information after applying PCA
print("\nAfter applying PCA:")
print("Number of features (dimensions) after PCA:", X_train_pca.shape[1])

# Visualizeing the first two principal components
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=40)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Components Visualization')
plt.colorbar()
plt.show()

models = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    SVC(probability=True)
]

# Storing evaluation metrics for each model for later viszulation(if needed)
metrics = []

for model in models:
    print(f"Training {type(model).__name__}")

    try:
        model.fit(X_train_pca, y_train)
        train_pred = model.predict(X_train_pca)
        test_pred = model.predict(X_test_pca)
        y_test_one_hot = np.eye(len(np.unique(y_test)))[y_test]

        model_name = type(model).__name__
        print(model_name)
        print("\n \n")

        print("****Train****")
        print("Accuracy:", accuracy_score(y_train, train_pred))
        
        # Calculating micro and macro averages for precision, recall, and F1 score
        precision_micro, precision_macro, _, _ = precision_recall_fscore_support(y_train, train_pred, average='micro', labels=np.unique(train_pred))
        recall_micro, recall_macro, _, _ = precision_recall_fscore_support(y_train, train_pred, average='micro', labels=np.unique(train_pred))
        f1_micro, f1_macro, _, _ = precision_recall_fscore_support(y_train, train_pred, average='micro', labels=np.unique(train_pred))

        print("Precision (Micro):", precision_micro)
        print("Precision (Macro):", precision_macro)
        print("Recall (Micro):", recall_micro)
        print("Recall (Macro):", recall_macro)
        print("F1 Score (Micro):", f1_micro)
        print("F1 Score (Macro):", f1_macro)

        print("Confusion Matrix:")
        print(confusion_matrix(y_train, train_pred))
        print("\n \n")

        print("****Test****")
        print("Accuracy:", accuracy_score(y_test, test_pred))
        
        # Calculating micro and macro averages for precision, recall, and F1 score
        precision_micro, precision_macro, _, _ = precision_recall_fscore_support(y_test, test_pred, average='micro', labels=np.unique(test_pred))
        recall_micro, recall_macro, _, _ = precision_recall_fscore_support(y_test, test_pred, average='micro', labels=np.unique(test_pred))
        f1_micro, f1_macro, _, _ = precision_recall_fscore_support(y_test, test_pred, average='micro', labels=np.unique(test_pred))

        print("Precision (Micro):", precision_micro)
        print("Precision (Macro):", precision_macro)
        print("Recall (Micro):", recall_micro)
        print("Recall (Macro):", recall_macro)
        print("F1 Score (Micro):", f1_micro)
        print("F1 Score (Macro):", f1_macro)

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, test_pred))

        # Ploting ROC-AUC curve
        fpr, tpr, _ = roc_curve(y_test_one_hot.ravel(), model.predict_proba(X_test_pca).ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.show()

        metrics.append({
            'Model': model_name,
            'Train Accuracy': accuracy_score(y_train, train_pred),
            'Test Accuracy': accuracy_score(y_test, test_pred),
            'Train Precision (Micro)': precision_micro,
            'Test Precision (Micro)': precision_micro,
            'Train Precision (Macro)': precision_macro,
            'Test Precision (Macro)': precision_macro,
            'Train Recall (Micro)': recall_micro,
            'Test Recall (Micro)': recall_micro,
            'Train Recall (Macro)': recall_macro,
            'Test Recall (Macro)': recall_macro,
            'Train F1 Score (Micro)': f1_micro,
            'Test F1 Score (Micro)': f1_micro,
            'Train F1 Score (Macro)': f1_macro,
            'Test F1 Score (Macro)': f1_macro,
            'AUC': roc_auc
        })

    except Exception as e:
        print(f"Training {type(model).__name__} failed with error: {e}")

# # Print evaluation metrics
# print("\nEvaluation Metrics:")
# print(metrics)


# In[ ]:


#component size : 30


import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# path to the directory containing the npz files
directory_path = 'C:/Users/AJAYK/Downloads/Kannada_MNIST_datataset_paper.zip/Kannada_MNIST_datataset_paper/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST'

# Loading the training and test data
X_train_data = np.load(os.path.join(directory_path, 'X_kannada_MNIST_train.npz'))
X_test_data = np.load(os.path.join(directory_path, 'X_kannada_MNIST_test.npz'))

# Accessing image data
X_train = X_train_data['arr_0']
X_test = X_test_data['arr_0']

# Loading the training and test labels
y_train_data = np.load(os.path.join(directory_path, 'y_kannada_MNIST_train.npz'))
y_test_data = np.load(os.path.join(directory_path, 'y_kannada_MNIST_test.npz'))

# Accessing  label data
y_train = y_train_data['arr_0']
y_test = y_test_data['arr_0']

# Normalizeing the pixel values in the image data
X_train_data_normalized = X_train / 255.0
X_test_data_normalized = X_test / 255.0

# Flattening the images after normalizing
X_train_split = X_train_data_normalized.reshape(X_train_data_normalized.shape[0], -1)
X_test_split = X_test_data_normalized.reshape(X_test_data_normalized.shape[0], -1)

X = X_train_split  # Flattened image data for training
y = y_train  # Training labels

X_test = X_test_split  # Flattened image data for testing
y_test = y_test  # Test labels

# Number of images in the training set
num_train_images = X_train.shape[0]

# Number of images in the test set
num_test_images = X_test.shape[0]

print("Number of images in the training set:", num_train_images)
print("Number of images in the test set:", num_test_images)
print("\n")

# Information about the original dataset
print("Original dataset dimensions:")
print("Number of features (pixels) per image:", X_train.shape[1])
print("Original number of features (dimensions):", X_train.shape[1] * X_train.shape[2])
print("\n")

# PCA to reduce dimensionality to 30 components
pca = PCA(n_components=30)
X_train_pca = pca.fit_transform(X)
X_test_pca = pca.transform(X_test)

#Information after applying PCA
print("\nAfter applying PCA:")
print("Number of features (dimensions) after PCA:", X_train_pca.shape[1])

# Visualizeing the first two principal components
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=40)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Components Visualization')
plt.colorbar()
plt.show()

models = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    SVC(probability=True)
]

# Storing evaluation metrics for each model for later viszulation(if needed)
metrics = []

for model in models:
    print(f"Training {type(model).__name__}")

    try:
        model.fit(X_train_pca, y_train)
        train_pred = model.predict(X_train_pca)
        test_pred = model.predict(X_test_pca)
        y_test_one_hot = np.eye(len(np.unique(y_test)))[y_test]

        model_name = type(model).__name__
        print(model_name)
        print("\n \n")

        print("****Train****")
        print("Accuracy:", accuracy_score(y_train, train_pred))
        
        # Calculating micro and macro averages for precision, recall, and F1 score
        precision_micro, precision_macro, _, _ = precision_recall_fscore_support(y_train, train_pred, average='micro', labels=np.unique(train_pred))
        recall_micro, recall_macro, _, _ = precision_recall_fscore_support(y_train, train_pred, average='micro', labels=np.unique(train_pred))
        f1_micro, f1_macro, _, _ = precision_recall_fscore_support(y_train, train_pred, average='micro', labels=np.unique(train_pred))

        print("Precision (Micro):", precision_micro)
        print("Precision (Macro):", precision_macro)
        print("Recall (Micro):", recall_micro)
        print("Recall (Macro):", recall_macro)
        print("F1 Score (Micro):", f1_micro)
        print("F1 Score (Macro):", f1_macro)

        print("Confusion Matrix:")
        print(confusion_matrix(y_train, train_pred))
        print("\n \n")

        print("****Test****")
        print("Accuracy:", accuracy_score(y_test, test_pred))
        
        # Calculating micro and macro averages for precision, recall, and F1 score
        precision_micro, precision_macro, _, _ = precision_recall_fscore_support(y_test, test_pred, average='micro', labels=np.unique(test_pred))
        recall_micro, recall_macro, _, _ = precision_recall_fscore_support(y_test, test_pred, average='micro', labels=np.unique(test_pred))
        f1_micro, f1_macro, _, _ = precision_recall_fscore_support(y_test, test_pred, average='micro', labels=np.unique(test_pred))

        print("Precision (Micro):", precision_micro)
        print("Precision (Macro):", precision_macro)
        print("Recall (Micro):", recall_micro)
        print("Recall (Macro):", recall_macro)
        print("F1 Score (Micro):", f1_micro)
        print("F1 Score (Macro):", f1_macro)

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, test_pred))

        # Ploting ROC-AUC curve
        fpr, tpr, _ = roc_curve(y_test_one_hot.ravel(), model.predict_proba(X_test_pca).ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.show()

        metrics.append({
            'Model': model_name,
            'Train Accuracy': accuracy_score(y_train, train_pred),
            'Test Accuracy': accuracy_score(y_test, test_pred),
            'Train Precision (Micro)': precision_micro,
            'Test Precision (Micro)': precision_micro,
            'Train Precision (Macro)': precision_macro,
            'Test Precision (Macro)': precision_macro,
            'Train Recall (Micro)': recall_micro,
            'Test Recall (Micro)': recall_micro,
            'Train Recall (Macro)': recall_macro,
            'Test Recall (Macro)': recall_macro,
            'Train F1 Score (Micro)': f1_micro,
            'Test F1 Score (Micro)': f1_micro,
            'Train F1 Score (Macro)': f1_macro,
            'Test F1 Score (Macro)': f1_macro,
            'AUC': roc_auc
        })

    except Exception as e:
        print(f"Training {type(model).__name__} failed with error: {e}")

# # Print evaluation metrics
# print("\nEvaluation Metrics:")
# print(metrics)


# In[ ]:




