from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score

import numpy as np
from sklearn.model_selection import train_test_split

from projectlib import readdata

percentage = 100

print("Silhouette Analysis using " + str(percentage) + "% of random selected  instances.")
frames, labels = readdata("../../data", "frames.csv", "labels.csv")

instances_set = []

for idv in frames.keys():
    for instance in frames[idv]:
        instances_set.append(instance[1:])

instances = np.array(instances_set)
if percentage != 100:
    X_rej, X, y_train, y_test = train_test_split(instances, range(len(instances)), test_size=float(percentage) / 100,
                                                 random_state=42)
else:
    X = instances

text = ""
range_n_clusters = range(77, 102)
dimX = 5
dimY = 5
font = {'fontsize': 8, 'fontweight': "normal", 'verticalalignment': 'baseline', 'horizontalalignment': 'center'}
f, axarr = plt.subplots(dimX, dimY)
k = 0
t = 0
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    axarr[k, t].set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    axarr[k, t].set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    output = "For n_clusters =" + str(n_clusters) + ", The average silhouette_score is :" + str(silhouette_avg)
    print(output)
    text += output
    text += "\n"
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        axarr[k, t].fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        # axarr[k, t].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    axarr[k, t].set_title("K = %d , AVGS = %f" % (n_clusters, silhouette_avg), font)
    if k == dimY - 1:
        axarr[k, t].set_xlabel("Coefficient", font)
    if t == 0:
        axarr[k, t].set_ylabel("Clusters", font)

    # The vertical line for average silhouette score of all the values
    axarr[k, t].axvline(x=silhouette_avg, color="red", linestyle="--")

    axarr[k, t].set_yticks([])  # Clear the yaxis labels / ticks
    axarr[k, t].set_xticks([-0.1, 0.5, 1])

    if t < dimY - 1:
        t += 1
    elif t == (dimY - 1):
        t = 0
        k += 1
    print k, "----", t

plt.suptitle("Silhouette analysis for KMeans clustering on " + str(percentage) + "% of instances", fontsize=10,
             fontweight='bold')
for z in range(0, dimX - 1):
    plt.setp([a.get_xticklabels() for a in axarr[z, :]], visible=False)

plt.setp([a.get_yticklabels() for a in axarr[:, dimY - 1]], visible=False)

with open("silhouette_analysis_results_final_best.txt", "wb") as f:
    f.write(text)
    f.close()
plt.show()
