import numpy as np
from sklearn import metrics

def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def read_seedword(seedword_path):
    with open(seedword_path, 'r') as f:
        return [l.replace('\n','').split(',') for l in f]

def indicify_seedword(seedwords, vocab, vocab_size, n_latent):
    gamma_prior = np.zeros((vocab_size, n_latent))
    word_indices = []
    for idx_topic, seed_topic in enumerate(seedwords):
        for idx_word, seed_word in enumerate(seed_topic):
            idx_vocab = vocab[seed_word]
            gamma_prior[idx_vocab, idx_topic] = 1.0  # V x K
            word_indices.append(idx_vocab)
    return gamma_prior, word_indices

def sort_values(dict):
    return list(zip(*sorted(dict.items(), key=lambda item: item[1])))[0]

def print_top_words(beta, id_vocab, n_top_words=30):
    print ('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        print(" ".join([id_vocab[j]
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
        print('**********')
    print ('---------------End of Topics------------------')

def classification_evaluate(y_pred, y_true, labels, show=True):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision, recall, f1_score, support = metrics.precision_recall_fscore_support(\
                                                     y_true=y_true, \
                                                     y_pred=y_pred, \
                                                     labels=[0, 1, 2],\
                                                     average=None)
    if show:
        print ("accuracy={}".format(accuracy))
        for idx, (iprecision, irecall, if1_score, isupport) \
            in enumerate(zip(precision, recall, f1_score, support)):
            print ("{}-{}".format(idx, labels[idx]))
            print ("precision={}, recall={}, f1_score={}, support={}"\
                  .format(iprecision, irecall, if1_score, isupport))

    return (accuracy, precision, recall, support, f1_score)

def classification_evaluate_dl(model, dl, n_latent, labels, show=True):
    avg_accuracy = 0.
    avg_precision = [0.] * n_latent
    avg_recall = [0.] * n_latent
    dl_support = [0] * n_latent
    avg_f1_score = [0.] * n_latent
    for x_,  y_ in dl:
        # Compute accuracy
        theta_ = model.topic_prop(x_)
        theta_ = np.argmax(theta_, axis=1)

        accuracy, precision, recall, support, f1_score = \
            classification_evaluate(theta_, y_, labels, show=False)
        avg_accuracy += accuracy / len(dl.ds) * dl.bs
        for i in range(n_latent):
            avg_precision[i] += precision[i] / len(dl.ds) * dl.bs
            avg_recall[i] += recall[i] / len(dl.ds) * dl.bs
            dl_support[i] += support[i]
            avg_f1_score[i] += f1_score[i] / len(dl.ds) * dl.bs
    if show:
        print ("dl.ds: ", len(dl.ds), "supports: ", np.sum(dl_support))
        print ("avg_accuracy", "=", "{:.9f}".format(avg_accuracy))
        for i in range(n_latent):
            print("avg_precision_{}".format(i), "=", "{:.9f}".format(avg_precision[i]),
                  "avg_recall_{}".format(i), "=", "{:.9f}".format(avg_recall[i]),
                  "dl_support_{}".format(i), "=", "{:d}".format(dl_support[i]),
                  "avg_f1_score_{}".format(i), "=", "{:.9f}".format(avg_f1_score[i]))

