import numpy as np
import random


def kmeans(X, K, maxLoops):
    rows, col = X.shape

    # Computing the Euclidian distance matrix for all samples
    edm = np.array([[np.linalg.norm(x - y) for y in X] for x in X])
    # Define first seeds
    centroidIndexes = select_first_seeds(X, edm, K, rows)

    # Find nearest centroid for each point
    clusters = np.zeros(rows, dtype=int)
    elementIndex = 0

    # For each line (each element) of the Euclidean Distance Matrix, we compare the distances to the centroids
    for element in edm:
        # Only use valid distances (the ones to the centroids)
        validDistances = element[np.array(centroidIndexes)]

        # Choose the shortest distance to the centroid
        minIndex = np.argmin(validDistances)

        # Assign each element to the nearest centroid
        clusters[elementIndex] = minIndex
        elementIndex = elementIndex + 1

    loop = 1
    newClusters = np.zeros(rows, dtype=int)
    centroids = X[centroidIndexes]
    errorSum = 0

    # In each loop, compute the following:
    #   -> Compare if the previous and actual clusters are the same (not in the first loop)
    #   -> Assign as the old clusters the new clusters
    #   -> Re-calculate the new centroids
    #   -> Get the euclidean distance from the elements to the centroids
    #   -> Re-assign each element to the new centroids

    while ((np.array_equal(newClusters, clusters) != True) and (loop < maxLoops)):

        # Assign old pointsInCluster
        if (loop > 1):
            clusters = newClusters
            newClusters = np.zeros(rows, dtype=int)
        loop = loop + 1

        # Re-calculate centroids
        centroids = _calculate_centers(X, centroids, clusters)

        # Calculate distances to new centroids
        newClusters, errorSum = _labels_inertia(X, centroids)

    return centroids, clusters, errorSum, loop


def select_first_seeds (np_df, edm, K, rows):

    # Selecting randomly the first cluster centroid and storing it in seeds_indexes
    centroidIndexes = []
    centroidIndex = np.random.randint(rows)
    centroidIndexes.append(centroidIndex)

    # Selecting the K seeds using a weighted probability distribution
    for i in range(1, K):
        # For each sample compute the squared distance to the seeds
        dist2 = np.array([min([edm[i][j] ** 2 for j in centroidIndexes]) for i in range(rows)])

        # Probability distribution proportional to the squared distance
        probs = dist2 / dist2.sum()

        # Cumulative probability
        cumprobs = probs.cumsum()

        # Select uniformly a random value between 0 and 1 in order to choose a point drawn with the desired probability distribution
        r = random.uniform(0, 1)
        for i, prob in enumerate(cumprobs):
            if r < prob:
                seed_index = i
                break
        # Store the seed index in a list.
        centroidIndexes.append(seed_index)
    return centroidIndexes


def _calculate_centers(X, cluster_centers, labels):
    new_cluster_centers = np.zeros(cluster_centers.shape, dtype=float)

    for clust_num in range(cluster_centers.shape[0]):
        new_cluster_centers[clust_num] = np.mean(X[labels == clust_num], axis=0)

    return new_cluster_centers


def _labels_inertia(X, centroids):
    """Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-modes algorithm.
    """
    inertia = 0
    labels = np.empty(X.shape[0], dtype='int64')

    for i, point in enumerate(X):
        diss = np.sum((centroids - point) ** 2, axis=1)
        clust = np.argmin(diss)
        labels[i] = clust
        inertia += diss[clust]

    return labels, inertia


class IKMeansMinusPlus(object):
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None
        # self.cluster_sizes_ = None
        self.inertia_ = None
        self.n_iter = None

    def fit(self, X):
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter = \
            kmeans(X, self.n_clusters, self.max_iter)

        return self

    def fit_predict(self, X):
        """Compute cluster centroids and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X), but more efficient
        """
        return self.fit(X).labels_

    def predict(self, X):
        assert self.cluster_centers_ is not None, "Model not yet fitted."
        return _labels_inertia(X, self.cluster_centers_)[0]
