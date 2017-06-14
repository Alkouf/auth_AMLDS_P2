import numpy as np
import cPickle as pickle
import projectlib as pjlib
from sklearn.metrics import silhouette_score
import time
import json


def calculate_medoid(cluster, distances):
    """
    From a set of videos (ids) find which one minimizes the average distance to the rest videos.
    Return the video id that is the medoid of the cluster

    :return: The id of the video that is the medoid of the cluster
    """
    minid = 0
    mindist = np.finfo(float).max

    for i in range(len(cluster)):
        temp_dist = 0
        for j in range(len(cluster)):
            temp_dist += distances[i, j]
        # print ids[i], temp_dist
        if temp_dist < mindist:
            mindist = temp_dist
            minid = cluster[i]
    return minid


def sort_to_clusters(medoids, distances):
    n_vids = distances.shape[0]
    clusters = [[] for i in range(len(medoids))]
    for i in range(n_vids):
        mindist = np.finfo(float).max
        minmed = 0
        for j in range(len(medoids)):
            if distances[i, medoids[j]] < mindist:
                mindist = distances[i, medoids[j]]
                minmed = j
        clusters[minmed].append(i)
    return clusters


def kmedoids(k, distances, iterations=400, random_state=1):
    n_vids = distances.shape[0]
    np.random.seed(random_state * k)
    medoids = np.random.choice(range(distances.shape[0]), k)

    clusters = sort_to_clusters(medoids, distances)  # vale ta video sta clusters, sumfwna me ta medoids

    for i in range(iterations):  # !!!
        medoids = [calculate_medoid(clusters[i], distances) for i in range(len(clusters))]
        clusters = sort_to_clusters(medoids, distances)  # vale ta video sta clusters, sumfwna me ta medoids

    return medoids, clusters


def silh_analysis(medoids, clusters, distances):
    clst = np.empty(shape=(distances.shape[0],), dtype=int)
    for i in range(len(medoids)):
        for j in range(len(clusters[i])):
            clst[clusters[i][j]] = i
    print clst
    silh_score = silhouette_score(distances, labels=clst, metric="precomputed")
    # print "silhouette score:", silh_score
    return silh_score


def analysis_on_k(distances):
    n_k = range(2, 50)
    sils = []

    for k in range(2, 50):
        medoids, clusters = kmedoids(k=k, distances=distances)
        sils.append(silh_analysis(medoids, clusters, distances))

    print zip(range(2, 50), sils)

    import matplotlib.pyplot as plt
    plt.plot(n_k, sils)
    plt.ylabel("Number of clusters")
    plt.xlabel("Silhouette score")
    plt.title("Silhouette analysis for k-medoids clustering")
    plt.text(6, .037, r'k=6, score=0.036')  # proxeiro
    plt.show()


def export_data(distances, k=6):
    frames, labels = pjlib.readdata("../../data", "frames.csv", "labels.csv")
    videoids = frames.keys()

    medoids, clusters = kmedoids(k, distances)

    print medoids
    print clusters

    train_data = dict()

    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            dist_from_medoids = []
            for m in range(k):
                dist_from_medoids.append(distances[medoids[m], clusters[i][j]])
            train_data[videoids[clusters[i][j]]] = dist_from_medoids

    with open("../../data/training_dict_" + str(k) + "_medoids.json", "wb") as f:
        json_file = json.dumps(train_data)
        f.write(json_file)
        f.close()


starting_time = time.time()
distances = pickle.load(open("../../data/Hausdorff.p", 'rb'))

# analysis_on_k(distances)
export_data(distances, k=6)

print "Duration: ", round(time.time() - starting_time, 3)
