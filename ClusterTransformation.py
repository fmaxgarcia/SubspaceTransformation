import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from scipy.stats import entropy

import matplotlib.pyplot as plt
import itertools

def create_synthetic_data(dimensions=2, shift=1.0, num_samples=128):
    means = np.ones( (dimensions,)) * shift
    cov = np.eye( dimensions )
    data = np.random.multivariate_normal(means, cov, num_samples)
    labels = [0 if data[i,0] > means[0] else 1 for i in range(num_samples)]
    return data, np.asarray(labels)

def _select_distance_cluster_mapping(labels, target_centroids, source_centroids):
    permutations = list(itertools.permutations(labels))
    mapping = permutations[0]
    max_distance = float("inf")
    for permutation in permutations:
        center_distance = 0.0
        for i in range(len(permutation)):
            target_centroid = target_centroids[i]
            source_centroid = source_centroids[ permutation[i] ]
            if np.linalg.norm(source_centroid - target_centroid) > center_distance:
                center_distance = np.linalg.norm(source_centroid - target_centroid)
            # center_distance += np.linalg.norm(source_centroid - target_centroid)

        if center_distance < max_distance:
            max_distance = center_distance
            mapping = permutation 

    return mapping


def _compute_distributions(X, Y):
    labels = np.unique(Y)
    distributions = []
    for i in range(labels.shape[0]):
        l = labels[i]
        data_indices = np.nonzero(Y == l)[0]
        data = X[data_indices]
        sum_features = np.sum(data, axis=0) + 1e-10
        dist = sum_features / np.sum(sum_features)
        distributions.append( dist )

    return distributions



def _select_kl_cluster_mapping(Xs, Ys, Xt, klabels, labels):
    cluster_s_distributions = _compute_distributions(Xs, Ys)
    cluster_t_distributions = _compute_distributions(Xt, klabels)
    permutations = list(itertools.permutations(labels))
    mapping = permutations[0]
    best_kl = float("inf")
    for permutation in permutations:
        sum_kl = 0.0
        for i in range(len(permutation)):
            source_dist = cluster_s_distributions[i]
            target_dist = cluster_t_distributions[ permutation[i] ]
            sum_kl += entropy(source_dist, target_dist)

        if sum_kl < best_kl:
            best_kl = sum_kl
            mapping = permutation

    return mapping

def _find_target_clusters(Xt, klabels, kcenters):
    for i in range(10):
        for cluster_id in range(kcenters.shape[0]):
            cluster_data_indices = np.nonzero(klabels == cluster_id)[0]
            cluster_data = Xt[cluster_data_indices]
            data_distance = np.linalg.norm(cluster_data - kcenters[cluster_id], axis=1)
            indices = np.argsort(data_distance)
            sorted_data = cluster_data[indices]
            kcenters[cluster_id]  = np.mean( sorted_data[:sorted_data.shape[0]/2] )
    
    for i in range(Xt.shape[0]):
        x = Xt[i]
        best_distance = float("inf")
        for center_id in range(kcenters.shape[0]):
            distance = np.linalg.norm(kcenters[center_id] - x)
            if distance < best_distance:
                best_distance = distance
                klabels[i] = center_id

    return kcenters, klabels


def _compute_supervised_centroids(X, Y):
    centers = []
    labels = np.unique(Y)
    for i in range(labels.shape[0]):
        l = labels[i]
        sum_data = 0.0
        count = 0
        for j in range(X.shape[0]):
            if Y[j] == l:
                sum_data += X[j]
                count += 1

        centers.append( sum_data / count)

    return np.asarray(centers)


def transform_target(Xs, Ys, Xt, Yt=None, visualize=False):

    centers1 = _compute_supervised_centroids(Xs, Ys)
    # init_centroids =  centers1 + 10000.0 * (centers1 - np.mean(centers1))
    component_vectors = centers1 - np.mean(centers1)
    init_centroids =  np.mean(Xt) + 1000.0 * component_vectors

    transformed_Xs = []
    for i in range(Xs.shape[0]):
        transformed_Xs.append( Xs[i] + init_centroids[Ys[i]])
    transformed_Xs = np.asarray(transformed_Xs)


    labels = np.unique(Ys)
    if Yt == None:
        # kmeans = KMeans(n_clusters=labels.shape[0], n_init=1, init=init_centroids)
        # kmeans.fit(Xt)
        # klabels = kmeans.labels_
        # centers2 = kmeans.cluster_centers_
        # centers2, klabels = _find_target_clusters(Xt, klabels, centers2)
        spectral = SpectralClustering(n_clusters=labels.shape[0])
        spectral.fit(Xt)
        klabels = spectral.labels_
        centers2 = []
        for i in range(labels.shape[0]):
            cluster_data_indices = np.nonzero(klabels == labels[i])[0]
            cluster_data = Xt[cluster_data_indices]
            centers2.append( np.mean(cluster_data, axis=0) )
        centers2 = np.asarray(centers2)
    else:
        klabels = Yt
        centers2 = _compute_supervised_centroids(Xt, Yt)

    transformed_Xt = []
    # alignment_mapping = _select_distance_cluster_mapping(labels, centers2, centers1)
    alignment_mapping = _select_kl_cluster_mapping(Xs, Ys, Xt, klabels, labels)
    for i in range(Xt.shape[0]):
        data_centroid_index = klabels[i]
        data_centroid = centers2[ data_centroid_index ]
        alignment_centroid = centers1[ alignment_mapping[data_centroid_index] ]

        
        transformed_Xt.append(Xt[i] - (data_centroid - alignment_centroid))
    transformed_Xt = np.asarray(transformed_Xt)

    if visualize == True:
        _visualize_data(Xs, Ys, Xt, transformed_Xt, klabels)

    return transformed_Xt, Xs

def _visualize_data(Xs, Ys, Xt, transformed_Xt, klabels):
    fig = plt.figure()
    ax = fig.add_subplot(211)

    colors = ["red" if l == 0 else "white" for l in Ys.tolist()]
    colors2 = ["blue" if l == 0 else "black" for l in klabels]

    colors.extend(colors2)

    X = np.vstack( (Xs, Xt) )
    Y = np.hstack( (Ys, klabels) )

    ax.scatter(X[:,0], X[:,1], c=colors)
    count = 0
    for xy in zip(X[:,0], X[:,1]):
        ax.annotate('%d' % Y[count], xy=xy, textcoords='data')
        count += 1

    ax2 = fig.add_subplot(212)
    X = np.vstack( (Xs, transformed_Xt) )
    ax2.scatter(X[:,0], X[:,1], c=colors)
    count = 0
    for xy in zip(X[:,0], X[:,1]):
        ax2.annotate('%d' % Y[count], xy=xy, textcoords='data')
        count += 1
    plt.show()



if __name__ == '__main__':
    X1, Y1 = create_synthetic_data(shift=1.0)
    X2, Y2 = create_synthetic_data(shift=10.0)


    newXt, newXs = transform_target(X1, Y1, X2, visualize=True)

