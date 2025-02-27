"""
.. module:: test

test
*************

:Description: test

    

:Authors: bejar
    

:Version: 

:Created on: 07/07/2014 11:12 

"""
__author__ = 'bejar'



# from Leader import Leader
# from GlobalKMeans import GlobalKMeans
from sklearn.datasets import load_iris, make_circles
import pylab as pl
from kemlglearn.datasets import make_blobs
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from kemlglearn.datasets import cluster_generator
# from kemlglearn.metrics import within_scatter_matrix_score, between_scatter_matrix_score, calinski_harabasz_score,\
#     zhao_chu_franti_score, scatter_matrices_scores, davies_bouldin_score, variation_of_information_score, \
#     jaccard_score, JaccardArandFolkes, folkes_mallow_score
# from kemlglearn.cluster.consensus import SimpleConsensusClustering
# from sklearn.metrics.cluster import normalized_mutual_info_score

# X, y_data = make_blobs(n_samples=[25, 200], n_features=2, centers=[[1,1], [0,0]], random_state=2, cluster_std=[0.1, 0.4])
# X , y_data= load_iris(return_X_y=True)

# X, y_data = make_circles(n_samples=1000, noise=0.5, random_state=4, factor=0.5)


# nc = 12
# _, X = cluster_generator(n_clusters=nc, sepval=0.01, numNonNoisy=15, numNoisy=3, rangeN=[50, 100])
# ld = Leader(radius=25.0)
#
# #print timeit.timeit(stmt='ld.fit(X)',setup=setup,number=10)
# ld.fit(X)
#
# # print ld.cluster_centers_.shape[0]
#
# fig = plt.figure()
#
# # ax = fig.gca(projection='3d')
# # pl.scatter(X[:, 1], X[:, 2], zs=X[:, 0], c=ld.labels_, s=25)
# #
# ax = fig.add_subplot(111)
# plt.scatter(X[:,0],X[:,1],c=ld.labels_)
#
# plt.show()

# gkm = GlobalKMeans(n_clusters=nc, algorithm='bagirov')
# gkm.fit(X)
# print DaviesBouldin(X, gkm.labels_)
# print scatter_matrices_scores(X, gkm.labels_, ['Inertia'])

#
# print gkm.cluster_centers_.shape[0], gkm.inertia_
# print within_scatter_matrix_score(X, gkm.labels_)
# print between_scatter_matrix_score(X, gkm.labels_)


# fig = plt.figure()
#
# # ax = fig.gca(projection='3d')
# # pl.scatter(X[:, 1], X[:, 2], zs=X[:, 0], c=gkm.labels_, s=25)
#
# ax = fig.add_subplot(111)
# plt.scatter(X[:, 0], X[:, 1], c=gkm.labels_)
#
# plt.show()
# for nc in range(2, 16):
#     km = KMeans(n_clusters=nc)
#     km.fit(X)
#     #print km.cluster_centers_.shape[0], km.inertia_
#     print nc, scatter_matrices_scores(X, km.labels_, ['CH', 'ZCF', 'Hartigan', 'Xu'])
#     print DaviesBouldin(X, km.labels_)

# fig = plt.figure()
#
# ax = fig.add_subplot(111)
# plt.scatter(X[:,0],X[:,1],c=km.labels_)
#
# plt.show()

# km = KMeans(n_clusters=3)
# km.fit(X)
#
# # print variation_of_information_score(km.labels_, y_data)
# # print normalized_mutual_info_score(km.labels_, y_data)
# print(jaccard_score(km.labels_, y_data))
# print(folkes_mallow_score(km.labels_, y_data))
# print(JaccardArandFolkes(km.labels_, y_data))

# print DaviesBouldin(X, km.labels_)
# print scatter_matrices_scores(X, km.labels_, ['Inertia'])

# fig = plt.figure()
#
# ax = fig.add_subplot(111)
# plt.scatter(X[:,0],X[:,1],c=km.labels_)
#
# plt.show()


# simple = SimpleConsensusClustering(n_clusters=nc, n_components=40)
# simple.fit(X)
#
# print DaviesBouldin(X, simple.labels_)
# print scatter_matrices_scores(X, simple.labels_, ['Inertia'])

# fig = plt.figure()
#
# ax = fig.add_subplot(111)
# plt.scatter(X[:,0],X[:,1],c=simple.labels_)
#
# plt.show()


# simple = SimpleConsensusClustering(n_clusters=nc, n_components=40, consensus2='spectral')
# simple.fit(X)
#
# print DaviesBouldin(X, simple.labels_)
# print scatter_matrices_scores(X, simple.labels_, ['Inertia'])
#

# fig = plt.figure()
#
# ax = fig.add_subplot(111)
# plt.scatter(X[:, 0], X[:, 1], c=y_data)
#
# plt.show()
# from numpy.random import normal
# import numpy as np
# sc1=75
# v1=0.1
# sc2=75
# v2=0.9
# X = np.zeros((sc1 + sc2, 2))
# X[0:sc1, 0] = normal(loc=0.0, scale=v1, size=sc1)
# X[0:sc1, 1] = normal(loc=0.0, scale=v2, size=sc1)
# X[sc1:, 0] = normal(loc=1, scale=v1, size=sc2)
# X[sc1:, 1] = normal(loc=0.0, scale=v2, size=sc2)
# dlabels = np.zeros(sc1 + sc2)
# dlabels[sc1:] = 1
#
# from .KMedoidsFlexible import KMedoidsFlexible
#
# km = KMedoidsFlexible(n_clusters=2)
#
# labels = km.fit_predict(X)
#
# fig = plt.figure()
#
# ax = fig.add_subplot(111)
# plt.scatter(X[:, 0], X[:, 1], c=labels)
# medoids = km.cluster_medoids_
#
# for i, m in enumerate(medoids):
#     plt.scatter(medoids[i, 0], medoids[i, 1], c=i, marker='x', s=200)
#
# plt.show()
#
# labels = km.predict(X)
#
# print(labels)


from IKMeansMinusPlus import IKMeansMinusPlus
import pandas as pd
from sklearn import metrics
import time
from sklearn.cluster import KMeans

df = pd.read_csv("C:/Users/Marc/Desktop/MAI/Q2/SEL/PW2/Data/Iris/iris.csv", header=None).values
X = df[:, :-1].astype(float)
Y = df[:, -1]

# df = pd.read_csv("C:/Users/Marc/Desktop/MAI/Q2/URL/Unsupervised/Option B/kemlglearn/kemlglearn/datasets/a3.txt", header=None, sep='   ').values
# X = df.astype(float)

# df = pd.read_csv("C:/Users/Marc/Desktop/MAI/Q2/URL/Unsupervised/Option B/kemlglearn/kemlglearn/datasets/lr/letter-recognition.data", header=None).values
# X = df[:, 1:].astype(float)

n_clusters = 3

ikmmp = IKMeansMinusPlus(n_clusters=n_clusters, max_iter=100)
km = KMeans(n_clusters=n_clusters, init='random', max_iter=100)
kmpp = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100)

print("----- Initialization -----")
t = time.time()
labels = ikmmp.fit_predict_ini(X)
t2 = time.time()
print("Inertia: {}; Iter: {};".format(ikmmp.inertia_, ikmmp.n_iter_))
print("Silhouette: {}; Time: {};".format(
                                            metrics.silhouette_score(X, labels),
                                            t2 - t
                                            ))
print("----- Initial K-means -----")
t = time.time()
labels = ikmmp.fit_predict_kmeans(X)
t2 = time.time()
print("Inertia: {}; Iter: {};".format(ikmmp.inertia_, ikmmp.n_iter_))
print("Silhouette: {}; Time: {};".format(
                                            metrics.silhouette_score(X, labels),
                                            t2 - t
                                            ))
print("------ I-K-means+- ------")
t = time.time()
labels = ikmmp.fit_predict(X)
t2 = time.time()
print("Inertia: {}; Iter: {};".format(ikmmp.inertia_, ikmmp.n_iter_))
print("Silhouette: {}; Time: {}".format(
                                            metrics.silhouette_score(X, labels),
                                            t2 - t
                                            ))

print("--- K-means (sklearn) ---")
t = time.time()
labels = km.fit_predict(X)
t2 = time.time()
print("Inertia: {}; Iter: {};".format(km.inertia_, km.n_iter_))
print("Silhouette: {}; Time: {}".format(
                                            metrics.silhouette_score(X, labels),
                                            t2 - t
                                            ))

print("-- K-means++ (sklearn) --")
t = time.time()
labels = kmpp.fit_predict(X)
t2 = time.time()
print("Inertia: {}; Iter: {};".format(kmpp.inertia_, kmpp.n_iter_))
print("Silhouette: {}; Time: {}".format(
                                            metrics.silhouette_score(X, labels),
                                            t2 - t
                                            ))


