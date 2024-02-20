import numpy as np
import sys
import main

dataset1=main.dataset1
dataset2=main.dataset2
dataset1 = dataset1.rename(columns={dataset1.columns[-1]: 'class'})
dataset2 = dataset2.rename(columns={dataset2.columns[-1]: 'class'})

class node:
    def __init__(self, attribute=None, split_point=None):
        self.attribute=attribute
        self.split_point=split_point
        self.left_subtree = None
        self.right_subtree = None

    # used for testing
    def __eq__(self, other):
        if not isinstance(other, node):
            return False
        return self.attribute==other.attribute and self.split_point==other.split_point and \
               self.left_subtree==other.left_subtree and self.right_subtree==other.right_subtree

# finds gini index for a dataframe
def gini(df):
    if df.empty:
        return 0
    class_frequencies=df['class'].value_counts(normalize=True)
    sum_of_squared_probabilities=sum(np.power(frequency,2) for frequency in class_frequencies)
    return 1-sum_of_squared_probabilities

# finds gini index for a node split into 2 partitions
def gini_split(df, left_subset, right_subset):
    return len(left_subset)/len(df) * gini(left_subset) + \
           len(right_subset)/len(df) * gini(right_subset)

# returns a node with the best attribute and split point
def optimal_node(df, attributes):
    best_gini_index=sys.float_info.max
    best_attribute=None
    best_split_point=None

    for attribute in attributes:

        # sorts in ascending order
        sorted_unique_values=sorted(set(df[attribute]))

        split_points = [(x + y) / 2 for x, y in zip(sorted_unique_values[:-1], sorted_unique_values[1:])]

        # adds minimum element from column of interest to the front of "split_points"
        split_points.insert(0, sorted_unique_values[0])

        # adds max element  from column of interest to the end of "split_points"
        split_points.append(sorted_unique_values[-1])

        for point in split_points:
            subset_left=df[df[attribute] <= point]
            subset_right=df[df[attribute] > point]
            gini_index=gini_split(df, subset_left, subset_right)

            if gini_index < best_gini_index:
                best_gini_index = gini_index
                best_attribute = attribute
                best_split_point = point

    return node(best_attribute, best_split_point)
