from code.lib.projectlib import make_train_set,scorer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score,make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from code.lib.Utils import g_a_p2

x_train, y_train = make_train_set("../../data", "training_dict_f_4.json", "labels.csv", weighted=True)

classifier = OneVsRestClassifier(LinearSVC())

print[x_train[0]]
print([y_train[0]])
print([y_train[1]])

m=cross_val_score(classifier,x_train,y_train,scoring=scorer(metric=g_a_p2),cv=10)
print(m)