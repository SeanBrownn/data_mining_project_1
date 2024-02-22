import numpy as np
import sys
import main

dataset1 = main.dataset1
dataset2 = main.dataset2
dataset1 = dataset1.rename(columns={dataset1.columns[-1]: 'class'})
dataset2 = dataset2.rename(columns={dataset2.columns[-1]: 'class'})


class node:
    def __init__(self, attribute=None, split_point=None, class_label=None):
        self.attribute = attribute
        self.split_point = split_point
        self.class_label = class_label
        self.left_subtree = None
        self.right_subtree = None

    # used for testing
    def __eq__(self, other):
        if not isinstance(other, node):
            return False
        left_subtrees_equal = self.left_subtree is None and other.left_subtree is None or \
                              self.left_subtree is not None and self.left_subtree == other.left_subtree
        right_subtrees_equal = self.right_subtree is None and other.right_subtree is None or \
                               self.right_subtree is not None and self.right_subtree == other.right_subtree
        other_variables_equal = self.attribute == other.attribute and self.split_point == other.split_point and \
                                self.class_label == other.class_label
        return left_subtrees_equal and right_subtrees_equal and other_variables_equal

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


# finds gini index for a dataframe
def gini(df):
    if df.empty:
        return 0
    class_frequencies = df['class'].value_counts(normalize=True)
    sum_of_squared_probabilities = sum(np.power(frequency, 2) for frequency in class_frequencies)
    return 1 - sum_of_squared_probabilities


# finds gini index for a dataframe split into 2 partitions
def gini_split(df, left_subset, right_subset):
    return len(left_subset) / len(df) * gini(left_subset) + \
           len(right_subset) / len(df) * gini(right_subset)


# returns a node with the best attribute and split point
# attributes is a list of columns in df that we want to look at
# assumes continuous data in df
# assumes 'class' is the last "attribute" in attributes
def optimal_node(df, attributes):
    best_gini_index = sys.float_info.max
    best_attribute = None
    best_split_point = None

    # class is always the last column but class is not an "attribute"
    for attribute in attributes[:-1]:
        # sorts in ascending order
        sorted_unique_values = sorted(set(df[attribute]))

        split_points = [(x + y) / 2 for x, y in zip(sorted_unique_values[:-1], sorted_unique_values[1:])]

        for point in split_points:
            subset_left = df[df[attribute] <= point]
            subset_right = df[df[attribute] > point]
            gini_index = gini_split(df, subset_left, subset_right)

            if gini_index < best_gini_index:
                best_gini_index = gini_index
                best_attribute = attribute
                best_split_point = point

    # if all data points have the same value for all attributes
    if best_attribute is None:
        best_attribute = attributes[0]

    return node(best_attribute, best_split_point)


# used to recursively build subtrees in hunt's algorithm
# most_common_class is the most common class in df
# best_attribute is the best attribute to split df on
# subset is the subset of records in df where df[best_attribute] <= or > the split point
def set_subtree(subset, attributes, best_attribute, most_common_class):
    if subset.empty:
        return node(None, None, most_common_class)
    attributes_copy = attributes.copy()  # uses copy so we don't modify original list
    attributes_copy.remove(best_attribute)
    return hunts(subset, attributes_copy)


# returns the decision tree for a dataframe with attribute list "attributes"
# assumes that df is not empty and that its data is continuous
# assumes that "attributes" is in list type and that 'class' is the last "attribute" in attributes
def hunts(df, attributes):
    if all(class_label == df['class'].iloc[0] for class_label in df['class']):
        return node(None, None, df['class'].iloc[0])

    most_common_class = df['class'].mode().iloc[0]

    # if class is the only attribute left
    if not attributes[:-1]:
        return node(None, None, most_common_class)

    tree = optimal_node(df, attributes)

    best_attribute = tree.attribute
    best_split_point = tree.split_point
    subset_left = df[df[best_attribute] <= best_split_point]
    subset_right = df[df[best_attribute] > best_split_point]

    # if all data points have the same value for best_attribute
    if len(set(df[best_attribute])) <= 1:
        return node(None, None, most_common_class)

    tree.left_subtree = set_subtree(subset_left, attributes, best_attribute, most_common_class)
    tree.right_subtree = set_subtree(subset_right, attributes, best_attribute, most_common_class)

    return tree


# tree is a node in the decision tree
# record is a feature vector (pandas series) that we want to classify. the values in the pandas series must have indices
# corresponding to the feature name
def classify(tree, record):
    if tree.class_label is not None:
        return tree.class_label

    if record[tree.attribute] <= tree.split_point:
        return classify(tree.left_subtree, record)

    return classify(tree.right_subtree, record)


# builds a decision tree from the labeled records, classifies the unlabeled records
def decision_tree(labeled_records, unlabeled_records):
    labeled_records = labeled_records.rename(columns={labeled_records.columns[-1]: 'class'})
    tree = hunts(labeled_records, labeled_records.columns.tolist())
    # classifies each row and adds result to a list
    return [classify(tree, unlabeled_records.iloc[i, :]) for i in range(len(unlabeled_records.index))]
