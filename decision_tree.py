import statistics
from collections import Counter

import numpy as np
import sys

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import main

dataset1 = main.dataset1
dataset2 = main.dataset2


class decision_tree_classifier:
    class tree:
        def __init__(self, attribute=None, split_point=None, class_label=None):
            self.attribute = attribute
            self.split_point = split_point
            self.class_label = class_label
            self.left_subtree = None
            self.right_subtree = None

        # used for testing
        def __eq__(self, other):
            if not isinstance(other, decision_tree_classifier.tree):
                return False
            left_subtrees_equal = self.left_subtree is None and other.left_subtree is None or \
                                  self.left_subtree is not None and self.left_subtree == other.left_subtree
            right_subtrees_equal = self.right_subtree is None and other.right_subtree is None or \
                                   self.right_subtree is not None and self.right_subtree == other.right_subtree
            other_variables_equal = self.attribute == other.attribute and self.split_point == other.split_point and \
                                    self.class_label == other.class_label
            return left_subtrees_equal and right_subtrees_equal and other_variables_equal

        # helper method for decision_tree_classifier.predict_point
        def predict_point_helper(self, record):
            if self.class_label is not None:
                return self.class_label

            if record[self.attribute] <= self.split_point:
                return self.left_subtree.predict_point_helper(record)

            return self.right_subtree.predict_point_helper(record)

        def print(self):
            string = ""
            if self.attribute is not None:
                string += ('attribute: ' + self.attribute)
            if self.split_point is not None:
                string += (' split point: ' + str(self.split_point))
            if self.class_label is not None:
                string += (' class: ' + str(self.class_label))
            if self.left_subtree is not None:
                string += ('left subtree: ' + self.left_subtree.print())
            if self.right_subtree is not None:
                string += ('right subtree: ' + self.right_subtree.print())
            return string

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.x_train = None
        self.y_train = None
        self.tree = None

    def fit(self, x_train, y_train):
        self.x_train=x_train
        self.y_train=y_train
        self.tree=self.build_tree(x_train, y_train, x_train.columns.tolist(), 0)

    def build_tree(self, x_train, y_train, attributes, depth):
        first_class=y_train.iloc[0]

        # labels of the current subset we are working with
        labels=decision_tree_classifier.subset_labels(x_train, y_train)

        if all(class_label == first_class for class_label in labels):
            return decision_tree_classifier.tree(class_label=first_class)

        most_common_class = statistics.multimode(labels)[0] # can handle multiple modes

        # if attributes is empty or we hit max depth
        if not attributes or (self.max_depth is not None and depth == self.max_depth):
            return decision_tree_classifier.tree(class_label=most_common_class)

        tree = self.optimal_node(x_train, y_train, attributes)

        best_attribute = tree.attribute
        best_split_point = tree.split_point
        subset_left = x_train[x_train[best_attribute] <= best_split_point]
        subset_right = x_train[x_train[best_attribute] > best_split_point]

        attributes.remove(best_attribute)

        # if all data points have the same value for best_attribute
        if len(set(x_train[best_attribute])) <= 1:
            return decision_tree_classifier.tree(class_label=most_common_class)

        tree.left_subtree = self.build_tree(subset_left, y_train, attributes, depth + 1)
        tree.right_subtree = self.build_tree(subset_right, y_train, attributes, depth + 1)

        return tree

    # given a subset, returns the corresponding indexes in y_train
    @staticmethod
    def subset_labels(subset, y_train):
        subset_indexes=subset.index.tolist()
        return [y_train[i] for i in subset_indexes]

    # finds gini index for a subset of the training data/labels
    @staticmethod
    def gini(x_train, y_train):
        if x_train.empty:
            return 0
        class_counts=Counter(y_train)
        class_frequencies = [count / len(y_train) for _, count in class_counts.items()]
        sum_of_squared_probabilities = sum(np.power(frequency, 2) for frequency in class_frequencies)
        return 1 - sum_of_squared_probabilities

    # finds gini index for a dataframe split into 2 partitions
    @staticmethod
    def gini_split(left_subset, right_subset, y_train):
        total_records=len(left_subset) + len(right_subset) # total number of records in parent node

        left_labels=decision_tree_classifier.subset_labels(left_subset, y_train)
        right_labels=decision_tree_classifier.subset_labels(right_subset, y_train)

        return len(left_subset) / total_records * decision_tree_classifier.gini(left_subset, left_labels) + \
               len(right_subset) / total_records * decision_tree_classifier.gini(right_subset, right_labels)

    # returns a tree node with the best attribute and split point
    # attributes is a list of columns in df that we want to look at
    # assumes continuous data in df
    @staticmethod
    def optimal_node(x_train, y_train, attributes):
        best_gini_index = sys.float_info.max
        best_attribute = None
        best_split_point = None

        for attribute in attributes:
            # sorts in ascending order
            sorted_unique_values = sorted(set(x_train[attribute]))

            split_points = [(x + y) / 2 for x, y in zip(sorted_unique_values[:-1], sorted_unique_values[1:])]

            for point in split_points:
                subset_left = x_train[x_train[attribute] <= point]
                subset_right = x_train[x_train[attribute] > point]
                gini_index = decision_tree_classifier.gini_split(subset_left, subset_right, y_train)

                if gini_index < best_gini_index:
                    best_gini_index = gini_index
                    best_attribute = attribute
                    best_split_point = point

        # if all data points have the same value for all attributes
        if best_attribute is None:
            best_attribute = attributes[0]

        return decision_tree_classifier.tree(attribute=best_attribute, split_point=best_split_point)

    # classifies an unlabeled record using the decision tree
    def predict_point(self, record):
        return self.tree.predict_point_helper(record)

    def predict(self, x_test):
        return [self.predict_point(x_test.iloc[i, :]) for i in range(len(x_test))]

def optimal_depth():
    dataset2 = main.dataset2
    dataset2.iloc[:, -1] = dataset2.iloc[:, -1].replace(0, -1)
    X = dataset2.iloc[:, :-1]
    y = dataset2.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)

    x_vals = []
    y_vals = []

    for depth in range(1, 11): # tests values 1-10
        classifier=decision_tree_classifier(depth)
        classifier.fit(x_train, y_train)
        predicted_labels=classifier.predict(x_test)
        error = 1-accuracy_score(y_test, predicted_labels)
        x_vals.append(depth)
        y_vals.append(error)

    plt.plot(x_vals, y_vals, marker='o')
    plt.scatter(x_vals, y_vals, color='red')
    plt.xticks(range(2,20,2))
    plt.xlabel('max depth')
    plt.ylabel('error rate')
    plt.title('error rate vs max depth of decision tree')
    plt.show()

# optimal_depth()