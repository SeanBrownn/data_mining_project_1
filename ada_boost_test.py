import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import ada_boost
import main

class MyTestCase(unittest.TestCase):
    def test_weighted_error(self):
        actual_labels = [-1, -1, 1, 1, 1]
        expected_labels = [1, -1, 1, -1, -1]
        weights = [0.1, 0.2, 0.3, 0.1, 0.3]
        self.assertEqual(0.1, ada_boost.ada_boost_classifier.weighted_error(expected_labels, actual_labels, weights))

    def test_ada_boost(self):
        training_data={
            'points': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
        data=pd.DataFrame(training_data)
        labels=[1, 1, 1, -1, -1, -1, -1, 1, 1, 1]

        classifier=ada_boost.ada_boost_classifier(3)
        classifier.fit(data, labels)
        predictions=classifier.predict(data)
        self.assertTrue(np.array_equal(predictions, labels))

    def test_decision_tree(self):
        # tests if we classify dataset2 correctly. doesn't pass b/c our decision tree implementation is different
        # from the package's
        dataset2 = main.dataset2
        dataset2_data = dataset2.iloc[:, :-1]
        dataset2_labels = dataset2.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(dataset2_data, dataset2_labels, test_size=0.1,
                                                            random_state=20)
        d2_train, d2_test = train_test_split(dataset2, test_size=0.1, random_state=20)

        dataset2_classifier = DecisionTreeClassifier(random_state=20)
        dataset2_classifier.fit(x_train, y_train)
        expected_labels = dataset2_classifier.predict(x_test)

        actual_labels = decision_tree.decision_tree(d2_train, d2_test)

        self.assertTrue(np.array_equal(expected_labels, actual_labels))

    # plots error rate as a function of k (number of nearest neighbors)
    def optimal_k():
        dataset2 = main.dataset2
        dataset2.iloc[:, -1] = dataset2.iloc[:, -1].replace(0, -1)
        X = dataset2.iloc[:, :-1]
        y = dataset2.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)
        d2_train, d2_test = train_test_split(dataset2, test_size=0.15, random_state=23)
        d2_test.reset_index(drop=True, inplace=True)

        x_vals = []
        y_vals = []

        for k in range(1, 21):  # tests k values 1-20
            predicted_labels = ada_boost_seed(d2_test, k, 23)
            error = 1 - accuracy_score(y_test, predicted_labels)
            x_vals.append(k)
            y_vals.append(error)

        plt.plot(x_vals, y_vals, marker='o')
        plt.scatter(x_vals, y_vals, color='red')
        plt.xticks(range(1, 21, 2))
        plt.xlabel('k')
        plt.ylabel('error rate')
        plt.title('error rate vs number of classifiers')
        plt.show()

    def test2():
        dataset2 = main.dataset2
        dataset2.iloc[:, -1] = dataset2.iloc[:, -1].replace(0, -1)
        X = dataset2.iloc[:, :-1]
        y = dataset2.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)
        classifier = ada_boost_classifier(5)
        classifier.fit(x_train, y_train)
        print(classifier.predict(x_test))

if __name__ == '__main__':
    unittest.main()
