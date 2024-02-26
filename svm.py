import numpy as np
from cvxopt import matrix, solvers

class SVM:
    def __init__(self, C=15):
        self.C = C  # Regularization parameter
        self.alphas = None  # Lagrange multiplier of each data point in training data
        self.w = None   # Weights
        self.b = None   # Bias

    # Fit a SVM classifier
    def fit(self, X, y):
        # Store the shape of the input X, for use in our matrix math
        n_samples, n_features = X.shape

        # Convert target variable into a column vector
        y = y.values.reshape(-1, 1) * 1

        # Scale each feature in X by its corresponding label
        X_dash = y * X

        # Calculates Hessian matrix
        H = np.dot(X_dash, X_dash.T) * 1

        #Converting into cvxopt format
        P = matrix(H)
        q = matrix(-np.ones((n_samples, 1)))
        G = matrix(np.vstack((np.eye(n_samples)*-1,np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))

        solvers.options['show_progress'] = False

        # Solve the matrix
        sol = solvers.qp(P, q, G, h, A, b)

        # Extract and reshape the optimized Lagrange multipliers
        self.alphas = np.array(sol['x']).reshape(-1, 1)

        # Update the weights
        self.w = ((y * self.alphas).T @ X).values.reshape(-1,1)

        S = (self.alphas > 1e-4).flatten()
        # Update the bias
        self.b = y[S] - np.dot(X[S], self.w)

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b[0])
    

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

print("Accuracy on dataset 1:", accuracy_score(y_test, predictions))

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

print("Accuracy on dataset 2:", accuracy_score(y_test, predictions))