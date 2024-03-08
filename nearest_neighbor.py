import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import main


class knn_classifier:
    def __init__(self, k=3): # makes k=3 by default
        self.k=k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train


    # predicts an unlabeled record using Euclidean distance
    def predict_point(self, record):
        distances = []
        # for every labeled point, find the distance to the unlabeled one we are looking at
        for i in range(len(self.x_train[0])):
            distances.append(math.dist(self.x_train.iloc[i, :], record))

        df = self.x_train.copy()
        df['distance'] = distances
        df['class']=self.y_train
        k_nearest_neighbors = df.sort_values(by='distance').head(self.k)
        k_nearest_neighbors['weight'] = np.power(k_nearest_neighbors['distance'], -2)

        # finds weighted sum for each class
        weighted_votes = k_nearest_neighbors.groupby('class')['weight'].sum()
        return weighted_votes.idxmax()

    # x_test is dataframe of unlabeled records
    def predict(self, x_test):
        # iterates through each row b/c each row is a point
        return [self.predict_point(x_test.iloc[i, :]) for i in range(len(x_test[0]))]

# plots error rate as a function of k (number of nearest neighbors)
def optimal_k():
    dataset2 = main.dataset2
    dataset2.iloc[:, -1] = dataset2.iloc[:, -1].replace(0, -1)
    X = dataset2.iloc[:, :-1]
    y = dataset2.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=23)

    x_vals = []
    y_vals = []

    for k in range(1, 21): # tests k values 1-20
        classifier=knn_classifier(k)
        classifier.fit(x_train, y_train)
        predicted_labels=classifier.predict(x_test)
        error = 1-accuracy_score(y_test, predicted_labels)
        x_vals.append(k)
        y_vals.append(error)

    plt.plot(x_vals, y_vals, marker='o')
    plt.scatter(x_vals, y_vals, color='red')
    plt.xticks(range(2,20,2))
    plt.xlabel('k')
    plt.ylabel('error rate')
    plt.title('error rate vs number of nearest neighbors')
    plt.show()

#optimal_k()