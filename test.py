
from code.lib.projectlib import readdata


frames,labels = readdata("data","frames.csv","labels.csv")


labels_aa = []

for key in labels.keys():
    for l in labels[key]:
        labels_aa.append(l)


print(len(labels_aa))
print(len(set(labels_aa)))