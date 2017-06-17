import time
from os.path import join
import csv
import json

import numpy as np
from code.lib.Utils import g_a_p2

from average_precision_calculator import AveragePrecisionCalculator


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


def make_train_set(path, features_videos, labels, weighted=True):
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
    x = []
    y = []
    for key in features_dict.keys():
        x.append(features_dict[key])
        y.append(labels_dict[key])
    x_train = np.array(x)
    y_train = np.array(y)
    if not weighted:
        x_train = x_train.astype(np.bool).astype(np.int)

    return x_train, y_train


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
        print gap
    return gap/iterations


'''''
def trainmodel(training_set_path, classifier, model_name, model_path="../models"):
    """
    Function trainmodel trains a new model using a given training set (training_set_path) and a predefined
    classifier. It saves the new model in the  model_name file.
    """
    print("Starting...")
    # Features extraction and preparation
    start_ex = time.time()
    train_vid_ids, train_bin_labels, train_mean_rgb, train_mean_audio = readdata(training_set_path)
    train_X = np.hstack((train_mean_rgb, train_mean_audio))
    end_ex = time.time()
    print("Total time for data extraction and preparation: ", end_ex - start_ex)
    # Model Learning
    start_train = time.time()
    classifier.fit(train_X, train_bin_labels)
    end_train = time.time()
    print("Total time for model training: ", end_train - start_train)
    # Save model
    joblib.dump(classifier, join(model_path, model_name))
    return True


def validatemodel(model_file_path, validation_set_dir_path, normalize=True, multilearn=False):
    """
    Function valiadatemodel loads a given model (model_path), predicts the labels of a given validation
    set (validation_set_dir_path) and calculates the mectiv Global Average Precision.
    The normalization is available for models that have been produced from algorithms like SVC.
    The multilearn variable should be True for models that created by algorithms of scikit-multilearn.
    """
    print("Starting...")
    # Features extraction and preparation
    start_ex = time.time()
    test_vid_ids, test_bin_labels, test_mean_rgb, test_mean_audio = readdata(validation_set_dir_path)
    test_X = np.hstack((test_mean_rgb, test_mean_audio))
    end_ex = time.time()
    print("Total time for data extraction and preparation: ", end_ex - start_ex)
    # Load Learning model
    start_load = time.time()
    model = joblib.load(model_file_path)
    end_load = time.time()
    print("Total time for loading model:", end_load - start_load)
    # Prediction Time
    start_prediction = time.time()
    predictions_proba = model.predict_proba(test_X)
    end_prediction = time.time()
    print("Total prediction time: ", end_prediction - start_prediction)
    if multilearn:
        metric = metriccalculation(predictions_proba.toarray(), test_bin_labels, normalize)
    else:
        metric = metriccalculation(predictions_proba, test_bin_labels, normalize)
    print("Global Average Precision :")
    return metric
'''''
