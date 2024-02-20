import math
import numpy as np


# takes dataframe of labeled records and dataframe of unlabeled
# uses euclidean distance
def nearest_neighbor(labeled_records, unlabeled_records, k):
    # excludes the last column since it is the class label
    # transposes so that each column is the coordinates of an unlabeled point
    unlabeled_points = unlabeled_records.iloc[:, :-1].T

    labels = []  # class labels for currently unlabeled points

    for record in unlabeled_points.columns:
        distances = []
        # for every labeled point, find the distance to the unlabeled one we are looking at
        for i in range(len(labeled_records[0])):
            distances.append(math.dist(labeled_records.iloc[i, :-1], unlabeled_points[record]))

        df = labeled_records.copy()
        df = df.rename(columns={df.columns[-1]: 'class'})
        df['distance'] = distances
        k_nearest_neighbors = df.sort_values(by='distance').head(k)
        k_nearest_neighbors['weight'] = np.power(k_nearest_neighbors['distance'], -2)

        # finds weighted sum for each class
        weighted_votes = k_nearest_neighbors.groupby('class')['weight'].sum()
        labels.append(weighted_votes.idxmax())

    return labels
