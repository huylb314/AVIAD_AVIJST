import numpy as np
from sklearn import metrics
import torch.nn.functional as F
import os

def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def read_seedword(seedword_path):
    with open(seedword_path, 'r') as f:
        return [l.replace('\n','').split(',') for l in f]

def sort_values(dict):
    return list(zip(*sorted(dict.items(), key=lambda item: item[1])))[0]

def print_gamma(gamma, vocab, seedwords):
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=False)
    for idx_topic, seed_topic in enumerate(seedwords):
        print ("idx topic: {}".format(idx_topic))
        for idx_word, seed_word in enumerate(seed_topic):
            idx_vocab = vocab[seed_word]
            print ("(topic-{} word-\"{}\" id-{}): gamma={}".format(idx_topic, seed_word, idx_vocab, gamma[idx_vocab]))

def calc_perp(model, dl, gamma_prior_batch):
    cost=[]
    np.set_printoptions(precision=10)
    np.set_printoptions(suppress=False)
    for x_,  y_ in dl:
        n_d = np.sum(x_)
        c = model.test(x_, gamma_prior_batch)
        cost.append(c/n_d)
    print ('The approximated perplexity is: ',(np.exp(np.mean(np.array(cost)))))

def print_top_words(epoch, weights, id_vocab, n_top_words, result, write, printout=True):
    if printout: print ('---------------Printing the Topics------------------')
    string_out = ""
    for weight in weights:
        for i in range(len(weight)):
            string_out += " ".join([id_vocab[j] for j in weight[i].argsort()[:-n_top_words - 1:-1]])
            string_out += "\n"
            if write:
                with open(os.path.join(result, "{}.txt".format(epoch)), 'w+') as fw:
                    fw.write(string_out)
    if printout: print (string_out)
    if printout: print ('---------------End of Topics------------------')

def classification_evaluate(y_pred, y_true, labels, show=True):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision, recall, f1_score, support = metrics.precision_recall_fscore_support(\
                                                     y_true=y_true, \
                                                     y_pred=y_pred, \
                                                     labels=range(len(labels)),\
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
    for xr_, xrlen_, x_, y_ in dl:
        # Compute accuracy
        recon, posterior_mean_theta, posterior_logvar_theta, posterior_mean_pi, posterior_logvar_pi = \
                        model(xr_, xrlen_, x_, y_, compute_loss=False)
        posterior_mean_pi = F.softmax(posterior_mean_pi, dim=-1)
        temp_x = posterior_mean_pi.argmax(-1).int().data.cpu().tolist()
        temp_y = y_.flatten().int().data.cpu().tolist()

        accuracy, precision, recall, support, f1_score = \
            classification_evaluate(temp_x, temp_y, labels, show=False)
        avg_accuracy += accuracy / len(dl.ds) * dl.sampler.bs
        for i in range(n_latent):
            avg_precision[i] += precision[i] / len(dl.ds) * dl.sampler.bs
            avg_recall[i] += recall[i] / len(dl.ds) * dl.sampler.bs
            dl_support[i] += support[i]
            avg_f1_score[i] += f1_score[i] / len(dl.ds) * dl.sampler.bs
    if show:
        print ("dl.ds: ", len(dl.ds), "supports: ", np.sum(dl_support))
        print ("avg_accuracy", "=", "{:.9f}".format(avg_accuracy))
        for i in range(n_latent):
            print("avg_precision_{}".format(i), "=", "{:.9f}".format(avg_precision[i]),
                  "avg_recall_{}".format(i), "=", "{:.9f}".format(avg_recall[i]),
                  "dl_support_{}".format(i), "=", "{:d}".format(dl_support[i]),
                  "avg_f1_score_{}".format(i), "=", "{:.9f}".format(avg_f1_score[i]))

