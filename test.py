from code.lib.projectlib import make_train_set
import numpy as np



X,Y=make_train_set("data","training_dict_f_91.json","labels.csv")

print(X[1])
print(X[2])
print(Y[1])
print(Y[2])

X,Y=make_train_set("data","training_dict_f_91.json","labels.csv",weighted=False)

print(X[1])

print(X[2])

print(Y[1])
print(Y[2])



"""
f= np.array([[0,1,0,0,0,34,17],[0,1,0,0,0,34,17]])

k=f.astype(np.bool)
l=f.astype(np.bool).astype(np.int)
print(f)
print(k)
print(l)
"""