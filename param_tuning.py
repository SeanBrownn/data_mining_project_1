import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import main, svm, nearest_neighbor
import matplotlib.pyplot as plt
import seaborn as sns

# Conduct hyperparameter tuning for the svm model
learning_rates = [0.001, 0.01, 0.1]
lambdas = [0.001, 0.01, 0.1]
epochs_list = [100, 500, 1000]
Cs = [0.1, 1, 10]

# Record the best accuracy and corresponding parameters for dataset1
best_accuracy = 0
best_params = {}

results = []

# Load the first dataset
data1 = main.dataset1

data1.iloc[:, -1] = data1.iloc[:, -1].replace(0, -1)

# Separate X and y
X = data1.iloc[:, :-1]
y = data1.iloc[:, -1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)

for lr in learning_rates:
    for lambda_param in lambdas:
        for epochs in epochs_list:
            for C in Cs:
                classifier = svm.SVM(learning_rate=lr, lambda_param=lambda_param, epochs=epochs, C=C)
                classifier.fit(pd.DataFrame(X_train), pd.Series(y_train))
                predictions = classifier.predict(pd.DataFrame(X_test))
                accuracy = accuracy_score(y_test, predictions)
                
                results.append({'learning_rate': lr, 'lambda_param': lambda_param, 'epochs': epochs, 'C': C, 'accuracy': accuracy})
                
                # Update the best parameters if the current model performs better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'learning_rate': lr, 'lambda_param': lambda_param, 'epochs': epochs, 'C': C}

# Convert results to DataFrame for easier analysis and plotting
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
sns.histplot(results_df['accuracy'], kde=True)
plt.title('Distribution of Accuracies')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

print(f"Best Accuracy for datset 1: {best_accuracy}")
print("Best Parameters for dataset 1:", best_params)
print()

# Record the best accuracy and corresponding parameters for dataset2
best_accuracy = 0
best_params = {}

results = []

# Load the second dataset
data2 = main.dataset2

data2.iloc[:, -1] = data2.iloc[:, -1].replace(0, -1)

# Separate X and y
X = data2.iloc[:, :-1]
y = data2.iloc[:, -1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)

for lr in learning_rates:
    for lambda_param in lambdas:
        for epochs in epochs_list:
            for C in Cs:
                classifier = svm.SVM(learning_rate=lr, lambda_param=lambda_param, epochs=epochs, C=C)
                classifier.fit(pd.DataFrame(X_train), pd.Series(y_train))
                predictions = classifier.predict(pd.DataFrame(X_test))
                accuracy = accuracy_score(y_test, predictions)

                results.append({'learning_rate': lr, 'lambda_param': lambda_param, 'epochs': epochs, 'C': C, 'accuracy': accuracy})
                
                # Update the best parameters if the current model performs better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'learning_rate': lr, 'lambda_param': lambda_param, 'epochs': epochs, 'C': C}
                

# Convert results to DataFrame for easier analysis and plotting
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
sns.histplot(results_df['accuracy'], kde=True)
plt.title('Distribution of Accuracies')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

print(f"Best Accuracy for datset 2: {best_accuracy}")
print("Best Parameters for dataset 2:", best_params)

# Parameter tuning for nearest neighbors

ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Record the best accuracy and corresponding parameters for dataset1
best_accuracy = 0
best_params = {}

results = []

# Load the first dataset
data1 = main.dataset1

data1.iloc[:, -1] = data1.iloc[:, -1].replace(0, -1)

# Separate X and y
X = data1.iloc[:, :-1]
y = data1.iloc[:, -1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)

for k in ks:
    classifier = nearest_neighbor.knn_classifier(k = k)
    classifier.fit(pd.DataFrame(X_train), pd.Series(y_train))
    predictions = classifier.predict(pd.DataFrame(X_test))
    accuracy = accuracy_score(y_test, predictions)
                    
    results.append({'k': k, 'accuracy': accuracy})
                    
    # Update the best parameters if the current model performs better
    if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'k': k}

# Convert results to DataFrame for easier analysis and plotting
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
plt.plot(results_df['k'], results_df['accuracy'], marker='o', linestyle='-', color='b')
plt.title('KNN Accuracy for Different Values of K')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.xticks(results_df['k'])
plt.grid(True)
plt.show()

print(f"Best Accuracy for datset 1: {best_accuracy}")
print("Best Parameters for dataset 1:", best_params)
print()