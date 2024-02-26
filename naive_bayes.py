import pandas as pd
import numpy as np
from math import sqrt, exp, pi

class NaiveBayesClassifier:
    def __init__(self):
        # Empty dictionary to store summaries (various information about classes)
        self.summaries = {}
        # And another dictionary where each key is a class label
        # and each value is the prior probability of that class
        self.class_probs = {}

    # Calculates the Gaussian probability distribution function for x
    def gaussian_probability(self, x, mean, stdev):
        # Formula taken from slides
        return (1 / (sqrt(2 * pi) * stdev)) * (exp(-((x - mean)**2 / (2 * stdev**2))))

    # Computes mean and standard deviation for each feature in each class
    # As well as overall class probabilities
    # Store the results in the fields for summaries and class_probs
    def summarize(self, X, y):
        # Calculate summary statistics for each of the classes
        for class_value in np.unique(y):
            rows = X[y == class_value]
            summaries = [(np.mean(rows[column]), np.std(rows[column])) for column in rows]
            self.summaries[class_value] = summaries
            self.class_probs[class_value] = float(len(rows)) / len(y)

    # Calculates the probability of a give data point belonging to each class
    # Uses the Gaussian probability density function for each feature's value
    def class_probabilities(self, data):
        # Dictionary to store probability of being in each class
        probabilities = {}

        # Loop through the items in the summaries field, using them to calculate the probability of
        # this row being in each class
        for class_value, class_summaries in self.summaries.items():
            probabilities[class_value] = self.class_probs[class_value]
            for i in range(len(class_summaries)):
                mean, stdev = class_summaries[i]
                probabilities[class_value] *= self.gaussian_probability(data[i], mean, stdev)
        return probabilities
    
    # Fit a model to the training data by calling the summarize function
    def fit(self, X, y):
        self.summarize(X, y)

    # Using our already defined model, predict the classes of a new set of inputs
    def predict(self, X):
        predictions = []

        # Go through each instance in x, and find which class it has the highest probability of being in
        for index, row in X.iterrows():
            row_as_list = row.values.tolist() # Convert series to list

            # Get the probabilities for each 
            probabilities = self.class_probabilities(row_as_list)

            # Set best_label and best_prob to None and -1, respectively
            # These are the values we will check against to see which class to classify the row in
            best_label, best_prob = None, -1
            for class_value, probability in probabilities.items():
                if best_label is None or probability > best_prob:
                    best_prob = probability
                    best_label = class_value

            # Append the prediction for the class this sample is most likely to belong to
            predictions.append(best_label)
        return predictions
    

# Now test the Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import main

# Complete testing on first dataset

# Load in the dataset
data1 = main.dataset1

# Separate X and y
X = data1.iloc[:, :-1]
y = data1.iloc[:, -1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)

# Create classifier and fit to training data
classifier = NaiveBayesClassifier()
classifier.fit(X_train, y_train)

# Use trained model to predict on testing data
predictions = classifier.predict(X_test)

# Use accuracy_score to calculate accuracy of predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy on dataset1:", accuracy)


# Complete testing on second dataset

# Load in the dataset
data2 = main.dataset2

# Replace the categorical column 4 with a numeric attribute
replacement_dict={'Present':1, 'Absent':0}
data2[4]=data2[4].replace(replacement_dict)

# Separate X and y
X = data2.iloc[:, :-1]
y = data2.iloc[:, -1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)

# Create classifier and fit to training data
classifier = NaiveBayesClassifier()
classifier.fit(X_train, y_train)

# Use trained model to predict on testing data
predictions = classifier.predict(X_test)

# Use accuracy_score to calculate accuracy of predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy on dataset2:", accuracy)