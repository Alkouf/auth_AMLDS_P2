import time
from code.lib.projectlib import make_train_set,metriccalculation
from sklearn.svm import LinearSVC,SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import make_scorer
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from code.lib.Utils import g_a_p2

x_train, y_train = make_train_set("../../data", "training_dict_f_91.json", "labels.csv", True)


mlb = MultiLabelBinarizer(sparse_output=False)
y_train = mlb.fit_transform(y_train)

algorithm = SVC(kernel='linear', probability=True)
ensemble = BaggingClassifier(algorithm)
imbalance = NearMiss(version=2)

pipeline = make_pipeline(imbalance, ensemble)
classifier = OneVsRestClassifier(algorithm)

# scorer = make_scorer(metriccalculation, greater_is_better=True)
#
# start = time.time()
# scores = cross_val_score(classifier, x_train, y_train, scoring=scorer, cv=5)
#
# stop = time.time()
# print scores

classifier.fit(x_train[:150], y_train[:150])
predictions = classifier.predict_proba(x_train[150:])
print g_a_p2(predictions, y_train[150:])


# for i in range(0,20):
#     for j in range(0, len(predictions[0])):
#         print predictions[i][j],
#     print
# print "-----"
#
# for i in range(0, 20):
#     for j in range(0, len(predictions[0])):
#         print y_train[i][j],
#     print



