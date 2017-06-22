import time
from code.lib.projectlib import make_train_set,metriccalculation, scorer, sort_by_frequency, last_index_of_freq
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skmultilearn.problem_transform import ClassifierChain

x_train, y_train = make_train_set("../../data", "training_dict_f_91.json", "labels.csv", second_features_videos=None, weighted=False)

mlb = MultiLabelBinarizer(sparse_output=False)
original_binlabels = mlb.fit_transform(y_train[:])
original_classes = list(mlb.classes_)
binlabels, classes, class_frequencies = sort_by_frequency(original_binlabels, original_classes)

# If a label have more than 5 occurences, it is considered
number_of_labels = last_index_of_freq(class_frequencies, 5)
y_train = binlabels[:, :number_of_labels]

# create classifier
algorithm = DecisionTreeClassifier()
ensemble = BaggingClassifier(algorithm, random_state=10)
classifier = ClassifierChain(ensemble)

# run cross validation to evaluate the classifier
start_ex = time.time()
m=cross_val_score(classifier,x_train,y_train,scoring=scorer(metric=metriccalculation),cv=10)
end_ex = time.time()

train_time = end_ex - start_ex
print m
print(m.mean())
print train_time

# return back to the initial labels dataset
y_train = binlabels
# run classifier and save the model
classifier.fit(x_train, y_train)
joblib.dump(classifier, "../../best_model.pkl")

