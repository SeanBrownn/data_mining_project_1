import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import main
import naive_bayes
import svm
import nearest_neighbor

random.seed(42)
np.random.seed(42)

def split(X):
    # Define folds as 10
    folds = 10

    # Randomly shuffle the data by shuffling the indices
    shuffled_indeces = X.index.to_list()
    random.shuffle(shuffled_indeces)
    X_shuffled = X.loc[shuffled_indeces]

    # Calculate size of each fold
    fold_size = len(X_shuffled) // folds

    # Create list to store folds
    folds_data = []

    # Split data into folds
    for i in range(folds):
        start_index = i * fold_size
        end_index = min((i + 1) * fold_size, len(X_shuffled))
        folds_data.append(X_shuffled.iloc[start_index:end_index])

    return folds_data

def kfold_cv(model, data):
    # Define folds as 10, for simplicity
    folds = 10

    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    folds_data = split(data)

    for i in range(folds):
        # Combine the folds to form training and testing sets
        train_folds = pd.concat([fold for j, fold in enumerate(folds_data) if j != i], axis=0)
        test_fold = folds_data[i]

        X_train, y_train = train_folds.iloc[:, :-1], train_folds.iloc[:, -1]
        X_test, y_test = test_fold.iloc[:, :-1], test_fold.iloc[:, -1]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        total_accuracy += accuracy_score(y_test, predictions)
        total_precision += precision_score(y_test, predictions, average='macro', zero_division=0)
        total_recall += recall_score(y_test, predictions, average='macro', zero_division=0)
        total_f1 += f1_score(y_test, predictions, average='macro', zero_division=0)
    
    average_accuracy = total_accuracy / folds
    average_precision = total_precision / folds
    average_recall = total_recall / folds
    average_f1 = total_f1 / folds

    return average_accuracy, average_precision, average_recall, average_f1

def print_performance_metrics(performances):
    print("Accuracy:", performances[0])
    print("Precision:", performances[1])
    print("Recall", performances[2])
    print("F1:", performances[3])

# Load the two datasets
data1 = main.dataset1
data2 = main.dataset2

# Create Naive Bayes Classifier and run performance on 2 datasets
naive_bayes = naive_bayes.NaiveBayesClassifier()
print("Naive Bayes on dataset1:")
naive_bayes_performance = kfold_cv(naive_bayes, data1)
print_performance_metrics(naive_bayes_performance)
print()
print("Naive Bayes on dataset2:")
naive_bayes_performance = kfold_cv(naive_bayes, data2)
print_performance_metrics(naive_bayes_performance)

print()

# Create SVM with optimal parameters for the first dataset
classifier = svm.SVM(learning_rate=0.001, lambda_param=0.001, epochs=1000, C=0.1)
print("SVM on dataset1:")
svm_performance = kfold_cv(classifier, data1)
print_performance_metrics(svm_performance)
print()

# Create SVM with optimal parameters for the second dataset
classifier = svm.SVM(learning_rate=0.001, lambda_param=0.001, epochs=100, C=10)
print("SVM on dataset2:")
svm_performance = kfold_cv(classifier, data2)
print_performance_metrics(svm_performance)

def kfold_cv_nearest_neighbor(data, k):
    folds = 10  # Define folds as 10, for simplicity

    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    folds_data = split(data)  # Use the split function provided to get the folds

    for i in range(folds):
        test_fold = folds_data[i]
        train_folds = pd.concat([fold for j, fold in enumerate(folds_data) if j != i], axis=0)

        predicted_labels = nearest_neighbor.nearest_neighbor(train_folds, test_fold, k)

        # Extract the true labels from the test fold
        true_labels = test_fold.iloc[:, -1].values

        # Calculate performance metrics
        total_accuracy += accuracy_score(true_labels, predicted_labels)
        total_precision += precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
        total_recall += recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
        total_f1 += f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    
    # Calculate average of each metric
    average_accuracy = total_accuracy / folds
    average_precision = total_precision / folds
    average_recall = total_recall / folds
    average_f1 = total_f1 / folds

    return average_accuracy, average_precision, average_recall, average_f1


print()
k_value = 3
print("Nearest Neighbors on dataset1:")
nn_performance = kfold_cv_nearest_neighbor(data1, k_value)
print_performance_metrics(nn_performance)
print()
print("Nearest Neighbors on dataset2:")
nn_performance = kfold_cv_nearest_neighbor(data2, k_value)
print_performance_metrics(nn_performance)
print()