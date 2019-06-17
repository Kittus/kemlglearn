import numpy as np
import random


def _unc_heuristic(point, centers):
    """Compute the heuristic value used to select a center, based on Useful Nearest Centers."""
    dis = np.linalg.norm(centers - point, axis=1)
    if 0 in dis:
        return -np.inf
    h = np.mean(dis) / np.max(dis) * np.sum(np.log(dis))
    return h


def _update_unc(X, UNC, new_center):
    """Update the Useful Nearest Centers for each point in the dataset when adding a new center."""
    for i, point in enumerate(X):
        new_useful = True
        remove_list = []
        d = np.linalg.norm(X[new_center] - point)

        for center_index in UNC[i]:
            d2 = np.linalg.norm(X[center_index] - point)
            c = np.linalg.norm(X[center_index] - X[new_center])

            # If the new center would render this one useless, save this information in the remove_list array
            if d < d2 and c < d2:
                remove_list.append(center_index)

            # If this center makes the new one useless for this point, mark it with the boolean
            elif d2 < d and c < d:
                new_useful = False

        if not new_useful:
            continue

        # If it is useful, add the element in the list and remove all those that now are useless
        UNC[i].append(new_center)
        for element in remove_list:
            UNC[i].remove(element)

    return UNC


def _unc_initialization(X, n_clusters):
    """Compute the whole initialization of cluster centers, returning the indexes of these center's points."""

    # Compute first center, which is useful nearest centers of all points
    cluster_indexes = [np.argmin(X[:, 0])]
    UNC = {}
    for i in range(X.shape[0]):
        UNC[i] = cluster_indexes.copy()

    # For each remaining center, iteratively select the one with the highest heuristic value, and update UNCs
    for _ in range(1, n_clusters):
        heuristics = np.array([_unc_heuristic(point, X[UNC[i]]) for i, point in enumerate(X)])
        new_center = np.nanargmax(heuristics)
        cluster_indexes.append(new_center)
        UNC =_update_unc(X, UNC, new_center)

    return cluster_indexes


def _calculate_centers(X, cluster_centers, labels):
    """Given associations of points to clusters, compute the new cluster centers as in classic k-means."""
    new_cluster_centers = np.zeros(cluster_centers.shape, dtype=float)

    for clust_num in range(cluster_centers.shape[0]):
        new_cluster_centers[clust_num] = np.mean(X[labels == clust_num], axis=0)

    return new_cluster_centers


def _labels_inertia(X, cluster_centers):
    """Calculate labels and the inertia (cost) functions given a matrix of points and
    a list of centroids for the k-means algorithm. Inertia is divided in clusters and second labels are computed.
    """
    inertias = np.zeros(cluster_centers.shape[0])
    labels = np.empty(X.shape[0], dtype='int64')
    second_labels = np.empty(X.shape[0], dtype='int64')

    for i, point in enumerate(X):
        diss = np.sum((cluster_centers - point) ** 2, axis=1)
        clusts = np.argsort(diss)
        labels[i] = clusts[0]
        second_labels[i] = clusts[1]
        inertias[clusts[0]] += diss[clusts[0]]

    return labels, second_labels, inertias


def kmeans_iter(X, labels, cluster_centers):
    """Single iteration of the k-means algorithm with divided inertias per cluster and second labels."""
    # Re-calculate centers with new labels
    cluster_centers = _calculate_centers(X, cluster_centers, labels)
    # Calculate new labels and inertia (assignation and cost) for the new centres
    new_labels, second_labels, inertias = _labels_inertia(X, cluster_centers)

    return cluster_centers, new_labels, second_labels, inertias


def kmeans(X, n_clusters, max_iter):
    """ Classic K-means algorithm with a Useful Nearest Centers initialization procedure. Given the number of clusters
        desired and the maximum amount of iterations, centers and labels are calculated iteratively.
        """
    # Initialize centers using the UNC approach
    cluster_indexes = _unc_initialization(X, n_clusters)
    cluster_centers = X[cluster_indexes]
    new_labels, second_labels, inertias = _labels_inertia(X, cluster_centers)
    n_iter = 0
    labels = np.ones(new_labels.shape)

    while np.array_equal(new_labels, labels) is False and n_iter < max_iter:
        # Update labels from previous iteration
        labels = new_labels
        # Compute the k-means iteration and increment iteration
        cluster_centers, new_labels, second_labels, inertias = kmeans_iter(X, labels, cluster_centers)
        n_iter += 1

    return cluster_centers, new_labels, second_labels, inertias, n_iter


def _cost(cluster_index, inertias, dist_sec):
    """Cost function heuristic that estimates the cost of removing the cluster."""
    return np.sum(dist_sec ** 2) - inertias[cluster_index]


def _gain(cluster_index, inertias):
    """Gain function heuristic that estimates the gain of dividing the cluster into two."""
    alfa = 3/4
    return alfa * inertias[cluster_index]


def _is_adjacent(cluster_index1, cluster_index2, labels, second_labels):
    """Given two clusters and the labels for the two closest centers to each point, determine if the first is adjacent
    to the second one. This means that an element assigned to the second has as second closest center the first one."""
    return cluster_index1 in second_labels[labels == cluster_index2]


def _is_strong_adjacent(cluster_index1, cluster_index2, labels, second_labels):
    """Given two clusters and labels for the two closest centers, determine if they are strong adjacent, this
    meaning one being adjacent to the other and the other way around."""
    return _is_adjacent(cluster_index1, cluster_index2, labels, second_labels) and \
           _is_adjacent(cluster_index2, cluster_index1, labels, second_labels)


def tkmeans(X, n_clusters, C_i, C_j, cluster_centers, labels, second_labels):
    """Modification to the classic k-means algorithm that only updates the clusters and labels of the points detected
    as active, so that it is more efficient. This can be applied in i-k-means-+ when eliminating and dividing clusters.
    The exact definition from the i-k-means-+ paper proposal has been followed.
    """
    # Step 1
    AC = {C_i, C_j}
    ACAdj = set([cluster for cluster in range(n_clusters) if _is_adjacent(cluster, C_j, labels, second_labels)])
    AP = set(np.array(range(X.shape[0]))[np.logical_or(labels == C_j, second_labels == C_j)])

    # Apply changes of centers
    cluster_centers[C_j] = random.choice(X[labels == C_i])
    labels, second_labels, inertias = _labels_inertia(X, cluster_centers)

    # Step 2
    while len(AC) > 0:
        for ac in AC:
            for cluster in range(n_clusters):
                if _is_adjacent(cluster, ac, labels, second_labels):
                    ACAdj.add(cluster)
            AP = AP.union(set(np.array(range(X.shape[0]))[np.logical_or(labels == ac, second_labels == ac)]))

        PotAC = set()

        # Step 3
        for ap in AP:
            diss = np.sum((cluster_centers - X[ap]) ** 2, axis=1)
            clusts = np.argsort(diss)
            clusts = np.array([clust for clust in clusts if clust in set(AC) or clust in set(ACAdj)])

            if labels[ap] != clusts[0]:
                PotAC.add(labels[ap])
                PotAC.add(clusts[0])
                labels[ap] = clusts[0]

            second_labels[ap] = clusts[1]

        # Step 4
        cluster_centers = _calculate_centers(X, cluster_centers, labels)

        # Step 5
        AC = PotAC
        AP = set()
        ACAdj = set()

    labels, second_labels, inertias = _labels_inertia(X, cluster_centers)
    return cluster_centers, labels, second_labels, inertias


def ikmeansminusplus(X, n_clusters, max_iter):
    """Main algorithm of the class. It works as an extention of k-means, by trying to improve the obtained results
    in a sort of post-processing stage where clusters are being removed and others divided in two parts. The initial
    k-means execution is initialized using UNC approach, and after that each iteration uses t-k-means instead.
    """

    # Produce first solution with classic k-means and UNC initialization
    cluster_centers, labels, second_labels, inertias, n_iter = kmeans(X, n_clusters, max_iter)

    # Variable initialization
    success = 0
    indivisible = [False] * n_clusters
    irremovable = [False] * n_clusters
    unmatchable = np.zeros([n_clusters, n_clusters], dtype=bool)

    # Repeat until end is reached
    finished = False

    while success <= n_clusters / 2 and not finished:
        # Select cluster S_i to divide
        S_i = None
        S_j = None

        while S_i is None:
            sorted_ind_gain = sorted(list(range(n_clusters)), key=lambda x: _gain(x, inertias), reverse=True)

            for pos, c_i in enumerate(sorted_ind_gain):
                if pos > (n_clusters - 1) / 2:
                    # k/2 clusters have a gain larger than S_i or no S_i available
                    return cluster_centers, labels, inertias, success
                if not indivisible[c_i]:
                    S_i = c_i
                    break

            if finished:
                break

            # Select cluster S_j to eliminate
            dists2 = np.linalg.norm(X - cluster_centers[second_labels], axis=1)
            sorted_ind_cost = sorted(list(range(n_clusters)), key=lambda x: _cost(x, inertias, dists2[labels == x]))
            goback = False

            for pos, c_i in enumerate(sorted_ind_cost):
                if pos > (n_clusters - 1) / 2:
                    goback = True
                if c_i != S_i and _cost(c_i, inertias, dists2[labels == c_i]) < _gain(S_i, inertias) \
                        and not unmatchable[S_i, c_i] and not _is_adjacent(S_i, c_i, labels, second_labels) \
                        and not _is_adjacent(c_i, S_i, labels, second_labels) and not irremovable[c_i]:
                    S_j = c_i
                    break

            if S_j is None:
                # No S_j available for this S_i
                return cluster_centers, labels, inertias, success
            if goback:
                # k/2 clusters have cost smaller than S_j
                indivisible[S_i] = True
                # Go back to S_i selection
                S_i = None
                break

        # Apply changes and t-k-means
        cluster_centers2, labels2, second_labels2, inertias2 = \
            tkmeans(X, n_clusters, S_i, S_j, cluster_centers, labels, second_labels)

        # Define consequences of the changes
        if sum(inertias2) > sum(inertias):
            unmatchable[S_i, S_j] = True
        else:
            irremovable[S_i] = True
            irremovable[S_j] = True

            for cluster in range(n_clusters):
                if _is_strong_adjacent(S_j, cluster, labels, second_labels):
                    indivisible[cluster] = True
                if _is_strong_adjacent(S_i, cluster, labels2, second_labels2) \
                        or _is_strong_adjacent(S_j, cluster, labels2, second_labels2):
                    irremovable[cluster] = True

            cluster_centers, labels, second_labels, inertias = cluster_centers2, labels2, second_labels2, inertias2
            success += 1

    return cluster_centers, labels, inertias, success


class IKMeansMinusPlus(object):

    """ I-K-means-+ clustering algorithm for numerical data.

    Parameters
    -----------
    n_clusters : int, optional, default: 3
        The number of clusters to form as well as the number of
        centers to generate.

    max_iter : int, optional, default: 100
        Maximum number of iterations of the initial k-means algorithm.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Feature-space points of the cluster's centers

    labels_ : array, [n_features]
        Labels of each point

    inertias_ : array, [n_clusters]
        Clustering inertia, defined as the sum distance of all points to
        their respective cluster centers, for each of the centers.

    inertia_ : float
        Clustering inertia, defined as the sum distance of all points to
        their respective cluster centers.

    n_iter_ : int
        Number of iterations used for the algorithm of the I-k-means-+

    """

    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.inertias_ = None
        self.n_iter_ = None

    def fit(self, X):
        """Compute i-k-means-+ clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]

        Returns
        -------
        self : this object so that the operation can be concatenated
        """
        self.cluster_centers_, self.labels_, self.inertias_, self.n_iter_ = \
            ikmeansminusplus(X, self.n_clusters, self.max_iter)
        self.inertia_ = sum(self.inertias_)

        return self

    def fit_predict(self, X):
        """Compute cluster centroids and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X), but more efficient
        """
        return self.fit(X).labels_

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        assert self.cluster_centers_ is not None, "Model not yet fitted."
        return _labels_inertia(X, self.cluster_centers_)[0]

    def fit_predict_ini(self, X):
        """ Fits and predicts but only using the UNC initialization stage of the algorithm.
            For testing purposes.
        """
        cluster_indexes = _unc_initialization(X, self.n_clusters)
        cluster_centers = X[cluster_indexes]
        self.labels_, _, self.inertias_ = _labels_inertia(X, cluster_centers)
        self.n_iter_ = 0
        self.inertia_ = sum(self.inertias_)

        return self.labels_

    def fit_predict_kmeans(self, X):
        """ Fits and predicts but only using the UNC initialization and initial k-means stage of the algorithm.
            For testing purposes.
        """
        self.cluster_centers_, self.labels_, _, self.inertias_, self.n_iter_ = \
            kmeans(X, self.n_clusters, self.max_iter)
        self.inertia_ = sum(self.inertias_)

        return self.labels_
