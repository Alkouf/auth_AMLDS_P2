"""
Clustering all instances of all bags (videos)
Number of instances: 46134  - Number of feature per instance: 1152
Number of videos: 200
Number of unique labels: 318
"""
print(__doc__)
from sklearn.cluster import KMeans
import json
import numpy as np

from projectlib import readdata


# Number of clusters/Number of generated features
n_clusters = 4

frames, labels = readdata("../../data", "frames.csv", "labels.csv")
videoids = frames.keys()
instances_set = []
instances_ids = {}
for idv in videoids:
    ids = []
    for instance in frames[idv]:
        instances_set.append(instance[1:])
        ids.append(instance[0])
    instances_ids[idv] = len(ids)

instances = np.array(instances_set)

# KMeans Train
clf = KMeans(n_clusters=n_clusters)
X = clf.fit_predict(instances)

# New features creation
train_dict={}
left_index=0

for idv in videoids:
    right_index=left_index+instances_ids[idv]
    train_dict[idv]=X[left_index:right_index]
    left_index+=instances_ids[idv]

training_set={}
for idv in train_dict.keys():
    features=[0]*n_clusters
    for cluster_num in train_dict[idv]:
        features[cluster_num]+=1
    training_set[idv]=features

print(len(training_set.keys()))
with open("../../data/training_dict_f_"+str(n_clusters)+".json","wb") as f:
    json_file = json.dumps(training_set)
    f.write(json_file)
    f.close()



