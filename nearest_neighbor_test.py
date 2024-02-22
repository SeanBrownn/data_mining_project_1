import main
import nearest_neighbor
import numpy as np
import unittest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# weight function used in our implementation of nearest neighbor
def weight_function(distances):
    return np.power(distances, -2)


class nearest_neighbor_test(unittest.TestCase):

    def test_nearest_neighbor(self):
        dataset1 = main.dataset1
        dataset1_data = dataset1.iloc[:, :-1]
        dataset1_labels = dataset1.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(dataset1_data, dataset1_labels, test_size=0.1,
                                                            random_state=13)
        d1_train, d1_test = train_test_split(dataset1, test_size=0.1, random_state=13)

        dataset_1_classifier = KNeighborsClassifier(n_neighbors=3, weights=weight_function)
        dataset_1_classifier.fit(x_train, y_train)
        expected_labels = dataset_1_classifier.predict(x_test)

        actual_labels = nearest_neighbor.nearest_neighbor(d1_train, d1_test, 3)

        self.assertEqual(set(actual_labels), set(expected_labels))
