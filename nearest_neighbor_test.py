import main
import nearest_neighbor
import numpy as np
import unittest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# weight function used in our implementation of nearest neighbor
def weight_function(distances):
    return np.power(distances, -2)


class MyTestCase(unittest.TestCase):

    def test_nearest_neighbor(self):
        dataset1 = main.dataset1
        dataset1_data = dataset1.iloc[:, :-1]
        dataset1_labels = dataset1.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(dataset1_data, dataset1_labels, test_size=0.1,
                                                            random_state=13)

        dataset1_classifier = KNeighborsClassifier(n_neighbors=3, weights=weight_function)
        dataset1_classifier.fit(x_train, y_train)
        expected_labels = dataset1_classifier.predict(x_test)

        knn_classifier=nearest_neighbor.knn_classifier() # k=3 by default
        knn_classifier.fit(x_train, y_train)
        actual_labels=knn_classifier.predict(x_test)

        self.assertTrue(np.array_equal(expected_labels, actual_labels))
