import math

import numpy as np
import random

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import main
import decision_tree

class ada_boost_classifier:

    # k = number of classifiers
    def __init__(self, k, decision_tree_depth=None):
        self.k=k
        self.classifiers=[] # list of decision tree classifiers
        self.alphas=[] # importance of each classifier
        self.x_train=None
        self.y_train=None
        self.decision_tree_depth=decision_tree_depth

    def fit(self, x_train, y_train):
        number_of_records = len(x_train)
        weights = [1/number_of_records] * number_of_records

        # each column is the class labels for a particular classifier
        labels_dataframe = pd.DataFrame()

        classifiers=[]

        for i in range(self.k):
            # samples with replacement based on weights
            selected_indices = random.choices(np.arange(number_of_records), weights=weights, k=number_of_records)
            training_data = x_train.iloc[selected_indices]
            training_labels=pd.Series(decision_tree.decision_tree_classifier.subset_labels(training_data, y_train))
            training_data.reset_index(drop=True, inplace=True)

            # builds a decision tree from training set, then returns class labels for the entire dataset
            classifier_i = decision_tree.decision_tree_classifier(self.decision_tree_depth)
            classifier_i.fit(training_data, training_labels)
            predictions=classifier_i.predict(x_train)

            error = ada_boost_classifier.weighted_error(y_train, predictions, weights)

            # resets weights, goes back to start of iteration i
            if error > 0.5:
                weights = [1 / number_of_records] * number_of_records
                i -= 1
                continue

            alpha_i = 0.5 * math.log((1 - error) / error)
            self.alphas.append(alpha_i)

            sum=np.sum(weights) # normalizes so that all weights sum to 1

            # since our classes are -1 and 1, the exponent is -alpha_i if the expected and actual labels agree, and
            # alpha_i if not

            weights=[weights[i] * np.exp(-alpha_i * y_train.reset_index(drop=True)[i] * predictions[i])/sum for i in range(len(weights))]
            #weights = weights * np.exp(-alpha_i * y_train * predictions) / np.sum(weights)
            self.classifiers.append(classifier_i)

    def predict(self, x_test):
        predictions=pd.DataFrame()
        for i in range(self.k):
            predictions[i]=(self.classifiers[i].predict(x_test))

        return np.sign(np.dot(predictions, self.alphas))

    @staticmethod
    def weighted_error(expected_labels, actual_labels, weights):
        incorrect = np.not_equal(expected_labels, actual_labels)
        error_sums = np.sum(weights * incorrect)
        return error_sums / len(weights)  # divide by number of records`

def optimal_k():
    dataset1 = main.dataset1
    dataset1.iloc[:, -1] = dataset1.iloc[:, -1].replace(0, -1)
    X = dataset1.iloc[:, :-1]
    y = dataset1.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)

    x_vals = []
    y_vals = []

    for k in range(6, 11): # tests k values 1-10
        classifier=ada_boost_classifier(k, 2)
        classifier.fit(x_train, y_train)
        predicted_labels=classifier.predict(x_test)
        error = 1-accuracy_score(y_test, predicted_labels)
        x_vals.append(k)
        y_vals.append(error)

    plt.plot(x_vals, y_vals, marker='o')
    plt.scatter(x_vals, y_vals, color='red')
    plt.xlabel('number of weak learners')
    plt.ylabel('error rate')
    plt.title('error rate vs number of weak learners')
    plt.show()

# optimal_k()