import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000, C=1):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.C = C
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        # Convert the inputs to numpy, used for dot product calculations later
        X = X.to_numpy()
        y = y.to_numpy()

        # Compute n and m, which will be used to initialize the weights numpy
        n, m = X.shape
        
        # Initialize the weights and bias to 0
        self.w = np.zeros(m)
        self.b = 0
        
        # Loop through the dataset 'epoch' times
        # Updating the weights using stochastic gradient descent
        for _ in range(self.epochs):
            # Loop through each row of X
            for i, x in enumerate(X):
                # Check if the sample would be classified as 1 
                classification = y[i] * (np.dot(x, self.w) - self.b) >= 1
            
                # Update weights and bias depending on which class the feature would be predicted in
                # Using the current weights and bias
                if classification:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - (np.dot(x, y[i])))
                    self.b -= self.lr * self.C * y[i]
                    
    def predict(self, X):
        X = X.to_numpy()
        activation_function = np.dot(X, self.w) - self.b
        return np.sign(activation_function)
   
# Quick test of SVM before cross validation
import main
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data1 = main.dataset1
data1.iloc[:, -1] = data1.iloc[:, -1].replace(0, -1)

# Separate X and y
X = data1.iloc[:, :-1]
y = data1.iloc[:, -1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)

# Create classifier and fit to training data
classifier = SVM()
classifier.fit(X_train, y_train)

# Use trained classifier to make predictions
predictions = classifier.predict(X_test)

# print("Accuracy on dataset 1:", accuracy_score(y_test, predictions))

# Test on dataset2
data2 = main.dataset2

data2.iloc[:, -1] = data2.iloc[:, -1].replace(0, -1)

# Separate X and y
X = data2.iloc[:, :-1]
y = data2.iloc[:, -1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)

# Create classifier and fit to training data
classifier = SVM()
classifier.fit(X_train, y_train)

# Use trained classifier to make predictions
predictions = classifier.predict(X_test)

# print("Accuracy on dataset 2:", accuracy_score(y_test, predictions))