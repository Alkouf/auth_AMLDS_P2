import time
from code.lib.projectlib import make_train_set,metriccalculation
from sklearn.svm import LinearSVC,SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import make_scorer
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
import code.lib.Utils as utl
import numpy as np
from sklearn.calibration import CalibratedClassifierCV


x_train, y_train = make_train_set("../../data", "training_dict_f_91.json", "labels.csv", True)

mlb = MultiLabelBinarizer(sparse_output=False)
original_binlabels = mlb.fit_transform(y_train[:])
original_classes = list(mlb.classes_)
binlabels, classes, class_frequencies = utl.sort_by_frequency(original_binlabels, original_classes)

truepos = [np.sum(binlabels[x]) for x in range(binlabels.shape[0])]
print truepos

# If a label have more than 10 occurences, it is considered
number_of_labels = utl.last_index_of_freq(class_frequencies, 15)

y_train = binlabels[:, :number_of_labels]
print y_train[:10]
print y_train.shape
print binlabels.shape

algorithm = SVC(kernel='linear', probability=True)
ensemble = BaggingClassifier(algorithm)
imbalance = NearMiss(version=2)
classifier = OneVsRestClassifier(CalibratedClassifierCV(ensemble, cv=2, method='isotonic'))  # sigmoid
# pipeline = make_pipeline(imbalance, ensemble)


# scorer = make_scorer(metriccalculation, greater_is_better=True)
#
# start = time.time()
# scores = cross_val_score(classifier, x_train, y_train, scoring=scorer, cv=5)
#
# stop = time.time()
# print scores

classifier.fit(x_train[:150], y_train[:150])
predictions = classifier.predict_proba(x_train[150:])
print utl.g_a_p2(predictions, y_train[150:], truepos)


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



