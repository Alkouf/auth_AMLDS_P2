import numpy as np
import average_precision_calculator as GAP
import os


def viz_frequencies(binlabels, classes, plot=False):
    """
    Given the binary label matrix, returns the frequencies of the labels, and plots the frequencies

    :param binlabels: The matrix that contains the labels, after multilabelBinarizer is applied
    :param classes: The list of the classes that correspond to the labels matrix
    :param plot: Boolean, if true, then a plot wil be generated and displayed.
    :return: The list of the frequencies of the labels, with the order they are on the classes list.
    """
    freq = []
    for i in range(binlabels.shape[1]):  # gia kathe label ston pinaka
        freq.append(np.sum(binlabels[:, i]))
    print "frequencies : "
    print freq
    if plot:
        import matplotlib.pyplot as plt
        print "plotting..."
        plt.plot(freq, classes)
        plt.ylabel('The number of cases that have the label')
        plt.xlabel('Labels')
        plt.show()
    return freq


def labels_min_occurences(binlabels, classes, minoccur=1):
    """
    Filters out the labels that are in less than 'minoccur' videos, and updates the classes list.
    :param binlabels: The original dense matrix of labels (after MultilabelBinarizer)
    :param classes: The list of the classes in binlabels.
    :param minoccur: The number of occurences that a label must be in order to preserved in the list.
    :return: Tuple, first the dense numpy matrix with labels (same form as MultilabelBinarizer), and secondly,
        the classes that remained after the filtering.
    """
    lb = np.zeros(shape=binlabels.shape, dtype=binlabels.dtype)
    cl = []
    index = 0
    rejected = 0

    for i in range(len(classes)):
        if np.sum(binlabels[:, i]) > minoccur:
            lb[:, index] = binlabels[:, i]
            index += 1
            cl.append(classes[i])
        else:
            rejected += 1

    print "Number of rejected labels: " + str(rejected) + " with minoccur: " + str(minoccur)
    return np.array(lb[:, 0:index]), list(cl)


def inverse_min_occurences(binlabels, classes, original_classes):
    """
    Expands the filtered matrix (see labels_min_occurences method), in order to be corresponding to the original set of
    classes.
    NOTE: If some classes have been filtered out, then the expanded matrix won't be exactly the same to the original.

    :param binlabels: The dense matrix with the filtered labels.
    :param classes: The classes in the binlabels matrix.
    :param original_classes: The original classes.
    :return:
    """
    r = binlabels.shape[0]
    c = len(original_classes)
    nc = len(classes)
    updated_labels = np.zeros(shape=(r, c), dtype=binlabels.dtype)
    cindex = 0
    d1 = 0
    for j in range(c):
        if cindex == nc:
            return updated_labels
        if original_classes[j] == classes[cindex]:
            for i in range(r):
                updated_labels[i][j] = binlabels[i][cindex]
            cindex += 1
            d1 += 1

    print "the number of common classes : " + str(cindex)

    return updated_labels


def g_a_p(predictions, Y_validation, positive_labels_count):
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
    gap = GAP.AveragePrecisionCalculator(20 * len(positive_labels_count))

    for i in range(valcases):
        gap.accumulate(predictions[i], Y_validation[i], num_positives=positive_labels_count[i])

    return gap.peek_ap_at_n()


def g_a_p2(predictions, Y_validation):
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

    for i in range(valcases):
        gap.accumulate(predictions[i], Y_validation[i], num_positives=np.sum(Y_validation[i]))

    return gap.peek_ap_at_n()


def appendToFile(fname, message):
    """
    Appends the message to the end of the file (fname). Also adds new line character after the message.

    :param fname: The name of the file.
    :param message: The string to be appended on the file.
    :return: Nothing
    """
    f = open(fname, 'a')
    message += str('\r\n')
    f.write(str(message))
    f.close()


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


def avg_matrices(m1, m2):
    """
    Calculates the average values element wise of the two given matrices.

    :param m1: Matrix 1
    :param m2: Matrix 2
    :return: numpy matrix of the same shape as m1, m2 that has the average element wise
    """
    if m1.shape != m2.shape:
        raise ValueError('Not equal shaped matrices!')
    return (np.array(m1) + np.array(m2)) / 2.


def export_predictions(predictions, classes, ids, csvname):
    """
    Exports the predictions to the file "csvname".
    The predictions are exported accordingly to Kaggle evaluation Format.

    :param predictions: The matrix with the predictions
    :param classes:
    :param ids:
    :param csvname:
    :return: None
    """
    if os.path.exists(csvname):
        os.remove(csvname)
    appendToFile(csvname, "VideoId,LabelConfidencePairs")
    print predictions.shape
    for i in range(predictions.shape[0]):
        pairs = zip(predictions[i], classes)
        pairs = sorted(pairs, key=lambda tup: tup[0], reverse=True)
        msg = str(ids[i]) + ","
        for j in range(20):
            msg += str(pairs[j][0]) + " "+str(pairs[j][1]) + " "
        appendToFile(csvname, msg)

