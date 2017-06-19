import time
from os.path import join
import csv
import json
from sklearn.metrics import make_scorer
import numpy as np
from code.lib.Utils import g_a_p2
from sklearn.preprocessing import MultiLabelBinarizer
from average_precision_calculator import AveragePrecisionCalculator
from sklearn.metrics import f1_score

def readdata(path, frames, labels):
    """
    The method read the data (frames, csv) and returns the respective dictionaries with VideoIds as keys.
    """
    with open(join(path, frames), "r") as f:
        reader = csv.reader(f)
        frames_list = list(reader)
        f.close()
    frames_dict = {}
    for row in frames_list:
        key = row[0]
        if key not in frames_dict.keys():
            concept = [np.array(row[1:])]
            frames_dict[key] = concept
        else:
            frames_dict[key].append(row[1:])

    with open(join(path, labels), "r") as f:
        reader = csv.reader(f)
        labels_list = list(reader)
        f.close()

    labels_dict = {}
    for row in labels_list:
        labels_dict[row[0]] = np.array(row[1:])

    return frames_dict, labels_dict


def metriccalculation(predictions, actual_labels, normalize=True):
    """
    Function metriccalculation calculates the Global Average Precision
    """
    start = time.time()
    calculator = AveragePrecisionCalculator(20)
    if normalize:
        for i in range(len(predictions)):
            calculator.accumulate(AveragePrecisionCalculator._zero_one_normalize(predictions[i]), actual_labels[i])
    else:
        for i in range(len(predictions)):
            calculator.accumulate(predictions[i], actual_labels[i])
    metric = calculator.peek_ap_at_n()
    end = time.time()
    print("Total time for Global Average Prediction: ", end - start)
    return metric


def make_train_set(path, features_videos, labels, second_features_videos=None, weighted=True):
    with open(join(path, labels), "r") as f:
        reader = csv.reader(f)
        labels_list = list(reader)
        f.close()
    labels_dict = {}
    for row in labels_list:
        labels_dict[row[0]] = np.array(row[1:])

    with open(join(path, features_videos), "rb") as f:
        features_dict = json.load(f)
        f.close()
    if not second_features_videos is None:
        with open(join(path, second_features_videos), "rb") as f:
            features_dict_medoids = json.load(f)
            f.close()
    x = []
    y = []
    for key in features_dict.keys():
        if not second_features_videos is None:
            x.append(np.hstack((features_dict[key], features_dict_medoids[key])))
        else:
            x.append(features_dict[key])
        y.append(labels_dict[key])
    x_train = np.array(x)
    y_train = np.array(y)
    y_train_bin = MultiLabelBinarizer().fit_transform(y_train)

    if not weighted:
        x_train = x_train.astype(np.bool).astype(np.int)

    return x_train, y_train_bin


def hold_out(classifier, data, labels, truepos = None, iterations=10, split=0.75):
    train_num = int(split*len(data))

    gap = 0.0
    for i in range(0, iterations):
        print "iteration: " + i.__str__()
        p = np.random.permutation(len(data))

        train_x = data[p][:train_num]
        train_y = labels[p][:train_num]

        test_x = data[p][train_num:]
        test_y = labels[p][train_num:]

        positives = truepos[p][train_num:]
        print np.shape(train_x),np.shape(train_y)
        classifier.fit(train_x, train_y)
        predictions = classifier.predict_proba(test_x)
        gap = gap + g_a_p2(predictions, test_y, positives)
        print g_a_p2(predictions, test_y, positives)
    return gap/iterations


def scorer():
    make_scorer(metriccalculation,needs_proba=True)