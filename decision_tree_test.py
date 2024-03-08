import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

import decision_tree
import main


class MyTestCase(unittest.TestCase):
    def test_gini(self):
        data1=pd.DataFrame()
        labels_1=['c2', 'c2', 'c2']
        self.assertEqual(0, decision_tree.decision_tree_classifier.gini(data1, labels_1))

        # only initialized so it isn't empty
        training_data = {
            'random': [3, 4, 5]
        }
        data2 = pd.DataFrame(training_data)
        self.assertEqual(0, decision_tree.decision_tree_classifier.gini(data2, labels_1))

        labels_2 = ['c1', 'c1', 'c2', 'c2', 'c2', 'c2']
        self.assertAlmostEqual(0.444, decision_tree.decision_tree_classifier.gini(data2, labels_2), delta=0.001)

    def test_gini_split(self):
        # only initialize left, right subsets so that they aren't empty when gini_split calls gini() on them
        left_child = {
            'random':[1,2,3,4,5,6]
        }
        left_subset = pd.DataFrame(left_child, index=([0,1,2,3,4,5]))
        right_child = {
            'random':[1,2,3,4,5,6]
        }

        # reset index so that right_subset.index works correctly
        right_subset = pd.DataFrame(right_child, index=([6,7,8,9,10,11]))
        labels=[1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
        self.assertAlmostEqual(0.361, decision_tree.decision_tree_classifier.gini_split(left_subset, right_subset, labels),
                               delta=0.001)

    def test_optimal_node(self):
        # only involves 1 attribute but tests ability to find best split point
        training_data = {
             'col': [60, 70, 75, 85, 90, 95, 100, 120, 125, 220]
        }
        labels_1 = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        data_1 = pd.DataFrame(training_data)
        tree = decision_tree.decision_tree_classifier.optimal_node(data_1, labels_1, data_1.columns)
        self.assertEqual(tree, decision_tree.decision_tree_classifier.tree('col', 97.5))

        training_data_2 = {
            'x': [1, 1, 0, 1],
            'y': [1, 1, 0, 0],
            'z': [1, 0, 1, 0]
        }
        labels_2 = [1, 1, 2, 2]
        data_2 = pd.DataFrame(training_data_2)
        tree2 = decision_tree.decision_tree_classifier.optimal_node(data_2, labels_2, data_2.columns)
        self.assertEqual(tree2, decision_tree.decision_tree_classifier.tree('y', 0.5))

    def setUp(self) -> None:
        # data used to test hunts and decision tree methods
        training_data = {
            'age': [24, 30, 36, 36, 42, 44, 46, 47, 47, 51],
            'likes dogs': [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
            'likes gravity': [0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
        }
        labels = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        data_1 = pd.DataFrame(training_data)

        self.full_classifier=decision_tree.decision_tree_classifier() # no max depth
        self.full_classifier.fit(data_1, labels)

        self.depth_2_classifier=decision_tree.decision_tree_classifier(2)
        self.depth_2_classifier.fit(data_1, labels)

    def test_fit(self):
        # tests tree with no max depth

        # begin process of building the expected tree from the leaves up
        yes_leaf = decision_tree.decision_tree_classifier.tree(class_label=1)
        no_leaf = decision_tree.decision_tree_classifier.tree(class_label=0)

        age_node = decision_tree.decision_tree_classifier.tree('age', 41.5, None)
        age_node.left_subtree = yes_leaf
        age_node.right_subtree = no_leaf

        likes_dogs_node = decision_tree.decision_tree_classifier.tree('likes dogs', 0.5, None)
        likes_dogs_node.left_subtree = age_node
        likes_dogs_node.right_subtree = yes_leaf

        # root of expected tree
        expected_tree = decision_tree.decision_tree_classifier.tree('likes gravity', 0.5, None)
        expected_tree.left_subtree = no_leaf
        expected_tree.right_subtree = likes_dogs_node

        self.assertTrue(self.full_classifier.tree.__eq__(expected_tree))

        # tests tree w/ max depth
        likes_dogs_node.left_subtree = yes_leaf
        self.assertTrue(self.depth_2_classifier.tree.__eq__(expected_tree))

    def test_predict_point(self):
        data = {
            'age': 28,
            'likes dogs': 1,
            'likes gravity': 0
        }
        record=pd.Series(data) # use pd.Series b/c this is the same type as each row of a dataframe
        self.assertEqual(self.full_classifier.predict_point(record), 0)

    def test_predict(self):
        unlabeled_records_data = {
            'age': [28, 45, 22, 36, 51, 29, 40, 24, 33, 55],
            'likes dogs': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
            'likes gravity': [0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
        }
        unlabeled_records = pd.DataFrame(unlabeled_records_data)
        self.assertEqual(self.full_classifier.predict(unlabeled_records), [0, 0, 1, 0, 1, 1, 0, 1, 0, 1])

if __name__ == '__main__':
    unittest.main()
