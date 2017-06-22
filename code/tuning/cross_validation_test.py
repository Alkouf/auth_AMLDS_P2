from code.lib.projectlib import make_train_set,scorer
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score,make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

x_train, y_train = make_train_set("../../data", "training_dict_f_4.json", "labels.csv", weighted=True)

classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))

print[x_train[0]]
print([y_train[0]])
print([y_train[1]])

m=cross_val_score(classifier,x_train,y_train,scoring=scorer(metric=g_a_p2),cv=10)
print(m)