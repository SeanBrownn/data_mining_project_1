import unittest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import decision_tree
import main


class MyTestCase(unittest.TestCase):
    def test_gini(self):
        table1 = {
            'class': ['c2', 'c2', 'c2']
        }
        df1 = pd.DataFrame(table1)
        self.assertEqual(0, decision_tree.gini(df1))

        table2 = {
            'class': ['c1', 'c1', 'c2', 'c2', 'c2', 'c2']
        }
        df2 = pd.DataFrame(table2)
        self.assertAlmostEqual(0.444, decision_tree.gini(df2), delta=0.001)

    def test_gini_split(self):
        parent = {
            'class': ['c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2', 'c2', 'c2']
        }
        df = pd.DataFrame(parent)

        left_child = {
            'class': ['c1', 'c1', 'c1', 'c1', 'c1', 'c2']
        }
        left_subset = pd.DataFrame(left_child)

        right_child = {
            'class': ['c1', 'c1', 'c2', 'c2', 'c2', 'c2']
        }
        right_subset = pd.DataFrame(right_child)
        self.assertAlmostEqual(0.361, decision_tree.gini_split(df, left_subset, right_subset),
                               delta=0.001)

    def test_optimal_node(self):
        # only involves 1 attribute but tests ability to find best split point
        data = {
            'value': [60, 70, 75, 85, 90, 95, 100, 120, 125, 220],
            'class': [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        }
        df = pd.DataFrame(data)
        tree = decision_tree.optimal_node(df, df.columns.tolist())
        self.assertEqual(tree, decision_tree.node('value', 97.5))

        data2 = {
            'x': [1, 1, 0, 1],
            'y': [1, 1, 0, 0],
            'z': [1, 0, 1, 0],
            'class': [1, 1, 2, 2]
        }
        df2 = pd.DataFrame(data2)
        tree2 = decision_tree.optimal_node(df2, df2.columns.tolist())
        self.assertEqual(tree2, decision_tree.node('y', 0.5))

    # data used to test hunts and decision tree methods
    data = {
        'age': [24, 30, 36, 36, 42, 44, 46, 47, 47, 51],
        'likes dogs': [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        'likes gravity': [0, 1, 1, 0, 0, 1, 0, 1, 1, 1],
        'class': [0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
    }
    labeled_records = pd.DataFrame(data)
    tree = decision_tree.hunts(labeled_records, labeled_records.columns.tolist())

    def test_hunts(self):
        # begin process of building the expected tree from the leaves up
        yes_leaf = decision_tree.node(None, None, 1)
        no_leaf = decision_tree.node(None, None, 0)

        age_node = decision_tree.node('age', 41.5, None)
        age_node.left_subtree = yes_leaf
        age_node.right_subtree = no_leaf

        likes_dogs_node = decision_tree.node('likes dogs', 0.5, None)
        likes_dogs_node.left_subtree = age_node
        likes_dogs_node.right_subtree = yes_leaf

        # root of expected tree
        expected_tree = decision_tree.node('likes gravity', 0.5, None)
        expected_tree.left_subtree = no_leaf
        expected_tree.right_subtree = likes_dogs_node

        self.assertEqual(MyTestCase.tree, expected_tree)

    def test_classify(self):
        data = {'age': 28, 'likes dogs': 1, 'likes gravity': 0}
        unlabeled_record = pd.Series(data)
        self.assertEqual(decision_tree.classify(MyTestCase.tree, unlabeled_record), 0)

    def test_decision_tree(self):
        unlabeled_records_data = {
            'age': [28, 45, 22, 36, 51, 29, 40, 24, 33, 55],
            'likes dogs': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
            'likes gravity': [0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
        }
        unlabeled_records = pd.DataFrame(unlabeled_records_data)
        self.assertEqual(decision_tree.decision_tree(MyTestCase.labeled_records, unlabeled_records),
                         [0, 0, 1, 0, 1, 1, 0, 1, 0, 1])

        # tests if we classify dataset2 correctly
        dataset2 = main.dataset2
        dataset2_data = dataset2.iloc[:, :-1]
        dataset2_labels = dataset2.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(dataset2_data, dataset2_labels, test_size=0.1,
                                                            random_state=20)
        d2_train, d2_test = train_test_split(dataset2, test_size=0.1, random_state=20)

        dataset_2_classifier = DecisionTreeClassifier(random_state=20)
        dataset_2_classifier.fit(x_train, y_train)
        expected_labels = dataset_2_classifier.predict(x_test)

        actual_labels = decision_tree.decision_tree(d2_train, d2_test)

        self.assertEqual(set(actual_labels), set(expected_labels))


if __name__ == '__main__':
    unittest.main()
