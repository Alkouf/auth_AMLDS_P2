import csv
from os.path import join
import numpy as np
from code.lib.projectlib import readdata


frames,labels = readdata("data","frames.csv","labels.csv")


print(len(frames.keys()))

print(len(labels.keys()))

keys= frames.keys()


print(len(frames[keys[0]][0]))
print(type(frames[keys[0]][0]))
print(frames[keys[0]][0])
print(frames[keys[1]][0])
