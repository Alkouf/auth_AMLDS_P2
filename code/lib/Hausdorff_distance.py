"""
This files exports (Hausdorff.p) an numpy array of size 200x200 that
contains the Hausdorff distances between all the videos for future usage on k medoids algorithm.
"""
from scipy.spatial.distance import directed_hausdorff
import projectlib as pjlib
import time
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import cPickle as pickle


def Hausdorff(A, B):
    """
    Calculates the Hausdorff distance between a pair of bags (videos), and returns the distance.
    """
    h1 = directed_hausdorff(A, B)[0]
    h2 = directed_hausdorff(B, A)[0]
    return max((h1, h2))


frames, labels = pjlib.readdata("../../data", "frames.csv", "labels.csv")
videoids = frames.keys()

starting_time = time.time()

print frames[videoids[0]][0]
print type(frames[videoids[0]][0][0])
A = np.array(frames[videoids[0]], dtype=float)
B = np.array(frames[videoids[1]], dtype=float)
print Hausdorff(A, B)

hd = np.empty(shape=(len(videoids), len(videoids)), dtype=float)

count = 0
for i in xrange(len(videoids)):
    for j in xrange(len(videoids)):
        hd[i, j] = Hausdorff(np.array(frames[videoids[i]], dtype=float), np.array(frames[videoids[j]], dtype=float))
        hd[j, i] = hd[i, j]
        count += 1
        if count % 100 == 0:
            print "Duration for", count, "pairs:", round(time.time() - starting_time, 3)

pickle.dump(hd, open("../../data/Hausdorff.p", 'wb'))

print "Duration:", round(time.time() - starting_time, 3)
