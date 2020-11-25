import numpy as np
from sklearn import metrics

def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def read_seedword(seedword_path):
    with open(seedword_path, 'r') as f:
        return [l.replace('\n','').split(',') for l in f]

def classification_evaluate(y_pred, y_true, labels, show=True):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision, recall, f1_score, support = metrics.precision_recall_fscore_support(\
                                                     y_true=y_true, \
                                                     y_pred=y_pred, \
                                                     labels=labels,\
                                                     average=None)
    if show:
        for idx, (iaccuracy, iprecision, irecall, if1_score, isupport) \
            in enumerate(zip(accuracy, precision, recall, f1_score, support)):
            print ("{}-{}".format(idx, labels[idx]))
            print ("accuracy={}, precision={}, recall={}, f1_score={}, support={}"\
                  .format(accuracy, iprecision, irecall, if1_score, isupport))

    return (accuracy, precision, recall, f1_score)