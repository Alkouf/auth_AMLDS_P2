import time
from os.path import join
import csv
import json
from sklearn.metrics import make_scorer
import numpy as np
from code.lib.Utils import g_a_p2
from sklearn.preprocessing import MultiLabelBinarizer
import average_precision_calculator as GAP
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



def make_train_set(path, features_videos, labels, second_features_videos=None, weighted=True):
    """
    Returns numpy arrays with the features (x_train), and numpy array with the binarized labels (y_train).
    The features are from the exported json files that are produced either from instance clustering (k-means)
    or from bag clustering (k-medoids).

    :param path: The path that contains the json files
    :param features_videos: The json files with the features
    :param labels: csv file with the labels
    :param second_features_videos: (optional) the second file with the features
    :param weighted: if true then it conserves the arithmetic values, otherwise just binary 1/0
    :return: the x_train and y_train from the data
    """
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

    return x_train, y_train


def sort_by_frequency(binlabels, classes):
    """
    Sorts the binlabel matrix (by the frequency on documents), and unpdates the corresponding classes list.
    Returns tuple, the sorted binlabels, and the corresponding classes list.

    :param binlabels: Dense matrix that has the binarized labels.
    :param classes: The list that describes the names of the binarized labels
    :return: Tuple, first the sorted binlabels as numpy array, and second the updated classes list.
    """
    f = np.zeros(shape=(binlabels.shape[1]), dtype=int)
    for i in range(binlabels.shape[1]):
        f[i] = np.sum(binlabels[:, i])
    c = zip(binlabels.T, classes)
    c = zip(f, c)
    c = sorted(c, key=lambda tup: tup[0], reverse=True)
    f, c = zip(*c)
    a, b = zip(*c)
    return np.array(a).T, list(b), list(f)


def last_index_of_freq(frequencies, target):
    """
    The frequencies is a list with integer values in descending order, and the method returs the index of the first
    item with value smaller than the target.

    :param frequencies: list of integer, sorted in decending order
    :param target: The min allowed value
    :return: the index of the last item that is larger or equal to the target value
    """
    index = 0
    for i in range(len(frequencies)):
        if frequencies[i] < target:
            return index
        index += 1
    return index

def metriccalculation(predictions, Y_validation, numpos=None):
    """
    Calculates the global average precision between the predictions and Y_validation arrays.

    :param predictions:
    :param Y_validation:
    :param positive_labels_count: In case the Y_validation array isn't the complete, this parameter gives the
        true number of positive labels.
    :return: The score given by GAP.
    """
    if predictions.shape != Y_validation.shape:
        raise ValueError("Different shapes between 'predictions' and 'Y_validation'")
    valcases = len(Y_validation)
    gap = GAP.AveragePrecisionCalculator(20 * valcases)

    predictions = np.array(predictions)
    Y_validation = np.array(Y_validation)

    for i in range(valcases):
        p = predictions[i].argsort()[::-1]
        predictions[i] = predictions[i][p]
        Y_validation[i] = Y_validation[i][p]
        if numpos is None:
            gap.accumulate(predictions[i][:20], Y_validation[i][:20], num_positives=np.sum(Y_validation[i]))
        else:
            gap.accumulate(predictions[i][:20], Y_validation[i][:20], num_positives=numpos[i])

    return gap.peek_ap_at_n()


def scorer(metric=metriccalculation):
    make_scorer(metric,needs_proba=True)