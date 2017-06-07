"""
Clustering all instances of all bags (videos)
Number of instances: 46134  - Number of feature per instance: 1152
Number of videos: 200
Number of unique labels: 318
"""
print(__doc__)
import numpy as np
from projectlib import readdata
from sklearn.cluster import KMeans



frames, labels = readdata("../../data", "frames.csv", "labels.csv")

videoids = frames.keys()
instances_set = []

for idv in videoids:
    for instance in frames[idv]:
        # print(instance)
        instances_set.append(instance[1:])

instances = np.array(instances_set)

