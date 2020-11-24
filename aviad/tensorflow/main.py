#!/usr/bin/python
import numpy as np
import tensorflow as tf
import itertools,time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt
import pickle
import sys, getopt
from models import prodlda, nvlda
from data_logger import DataLogger
from sklearn import metrics
import preprocess_aspects_analysis as pre_AA
import constants_aspects_analysis as constants_AA
import seedwords_helper as SW_helper
from sklearn.model_selection import StratifiedShuffleSplit

dataLoggerUtil = DataLogger()

# DATA Setting
PREPROCESS_DATA = False
if (PREPROCESS_DATA == True):    pre_AA.data_process()

'''-----------Data--------------'''
def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

data_onehot_location = constants_AA.const.TRAIN_FILE_LOCATION
data_onehot = np.load(data_onehot_location)

vocab_location = constants_AA.const.VOCAB_FILE_LOCATION
vocab = dataLoggerUtil.unpickle(vocab_location)
vocab_size=len(vocab)

#--------------convert to one-hot representation------------------
data = data_onehot[:,0]
label = data_onehot[:,1]

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
_, index_train_test = sss.split(data, label)

data_tr, label_tr = data[index_train_test[0]], label[index_train_test[0]]
data_te, label_te = data[index_train_test[1]], label[index_train_test[1]]

#--------------print the data dimentions--------------------------
print ('Data Loaded')
print ('Dim Training Data',data_tr.shape[0], vocab_size)
print ('Dim Test Data',data_te.shape[0], vocab_size)
'''-----------------------------'''

'''--------------Global Params---------------'''
n_samples_tr = data_tr.shape[0]
n_samples_te = data_te.shape[0]
docs_tr = data_tr
docs_te = data_te
batch_size=200
learning_rate=0.002
network_architecture = \
    dict(n_hidden_recog_1=100, # 1st layer encoder neurons
         n_hidden_recog_2=100, # 2nd layer encoder neurons
         n_hidden_gener_1=vocab_size, # 1st layer decoder neurons
         n_input=vocab_size, # MNIST data input (img shape: 28*28)
         n_z=50)  # dimensionality of latent space
data_prior = ()
'''-----------------------------'''

'''--------------Netowrk Architecture and settings---------------'''
def make_network(layer1=100,layer2=100,num_topics=50,bs=200,eta=0.002):
    tf.reset_default_graph()
    network_architecture = \
        dict(n_hidden_recog_1=layer1, # 1st layer encoder neurons
             n_hidden_recog_2=layer2, # 2nd layer encoder neurons
             n_hidden_gener_1=vocab_size, # 1st layer decoder neurons
             n_input=vocab_size, # MNIST data input (img shape: 28*28)
             n_z=num_topics)  # dimensionality of latent space
    batch_size=bs
    learning_rate=eta
    return network_architecture,batch_size,learning_rate

'''--------------Methods--------------'''
def compute_accuracy(y_pred, y_true, a_list_label):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_true=y_true, \
                                                     y_pred=y_pred, \
                                                     labels=a_list_label,\
                                                     average=None)

    return (accuracy, precision, recall, f1_score)

def train(network_architecture, type='prodlda',learning_rate=0.001,
          batch_size=200, training_epochs=100, display_step=5):
    tf.reset_default_graph()
    vae=''

    if type=='prodlda':
        vae = prodlda.VAE(network_architecture,
                                     learning_rate=learning_rate,
                                     batch_size=batch_size, data_prior=data_prior)
    elif type=='nvlda':
        vae = nvlda.VAE(network_architecture,
                                     learning_rate=learning_rate,
                                     batch_size=batch_size)
    emb=0

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_accuracy = 0.
        avg_precision = [0.] * network_architecture['n_z']
        avg_recall = [0.] * network_architecture['n_z']
        avg_f1_score = [0.] * network_architecture['n_z']

        sum_t_c = 0.

        batches_tr = n_samples_tr // batch_size
        # Loop over all batches
        for index_batch_tr in range(batches_tr):
            batch_data = data_tr[index_batch_tr*batch_size:(index_batch_tr+1)*batch_size]
            batch_label = label_tr[index_batch_tr*batch_size:(index_batch_tr+1)*batch_size]
            batch_data = np.array([onehot(doc,vocab_size) for doc in batch_data])
            
            emb = None
            t_c = time()
            cost,emb = vae.partial_fit(batch_data)
            c_elap = time() - t_c

            # Compute average loss
            avg_cost += cost / n_samples_tr * batch_size

            # Compute avg time
            sum_t_c += c_elap

            # Compute accuracy
            temp_theta = vae.topic_prop(batch_data)
            temp_theta_max_idx = np.argmax(temp_theta, axis=1)
            temp_theta_y_pred = np.expand_dims(temp_theta_max_idx, axis=1)

            batch_label = np.expand_dims(batch_label, axis=1)

            # to list
            temp_theta_y_pred = temp_theta_y_pred.tolist()
            batch_label = batch_label.tolist()
            
            accuracy, precision, recall, f1_score = compute_accuracy(temp_theta_y_pred, batch_label,\
                                                                     constants_AA.const.LIST_LABEL)
            avg_accuracy += accuracy / n_samples_tr * batch_size

            for k in range(network_architecture['n_z']):
               avg_precision[k] += precision[k] / n_samples_tr * batch_size
               avg_recall[k] += recall[k] / n_samples_tr * batch_size
               avg_f1_score[k] += f1_score[k] / n_samples_tr * batch_size

            if np.isnan(avg_cost):
               print (epoch, i, np.sum(batch_xs,1).astype(np.int), batch_xs.shape)
               print ('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
               # return vae,emb
               sys.exit()

        # Compute accuracy test
        batches_te = n_samples_te // batch_size
        theta_y_pred_te = []
        theta_label_te = []
        for index_batch_te in range(batches_te):
            batch_data_te = data_te[index_batch_te*batch_size:(index_batch_te+1)*batch_size]
            batch_label_te = label_te[index_batch_te*batch_size:(index_batch_te+1)*batch_size]
            batch_data_te = np.array([onehot(doc,vocab_size) for doc in batch_data_te])
            
            temp_theta_te = vae.topic_prop(batch_data_te)
            temp_theta_max_idx_te = np.argmax(temp_theta_te, axis=1)
            temp_theta_y_pred_te = np.expand_dims(temp_theta_max_idx_te, axis=1)

            theta_y_pred_te.extend(temp_theta_y_pred_te)
            theta_label_te.extend(batch_label_te)
            

        accuracy_te, precision_te, recall_te, f1_score_te = compute_accuracy(theta_y_pred_te, theta_label_te, \
                                                                             constants_AA.const.LIST_LABEL)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("##################################################", \
                 "\n", \
                 "Epoch:", '%04d' % (epoch+1), \
                 "\n", \
                 "cost=", "{:.9f}".format(avg_cost), \
                 "avg_accuracy=", "{:.9f}".format(avg_accuracy), \
                 "accuracy_te=", "{:.9f}".format(accuracy_te), \
                 "total_calculate=", "{:.4f}".format(sum_t_c))

            for k in range(network_architecture['n_z']):
               print ("avg_precision_{}".format(k), "=" , "{:.9f}".format(avg_precision[k]), \
                     "avg_recall_{}".format(k), "=" , "{:.9f}".format(avg_recall[k]), \
                     "avg_f1_score_{}".format(k), "=" , "{:.9f}".format(avg_f1_score[k]))
               print ("precision_te{}".format(k), "=" , "{:.9f}".format(precision_te[k]), \
                     "recall_te{}".format(k), "=" , "{:.9f}".format(recall_te[k]), \
                     "f1_score_te{}".format(k), "=" , "{:.9f}".format(f1_score_te[k]))
            # calcPerp(vae)
            print ("##################################################")

        write_topwords("results/ursa/{}.txt".format(epoch), emb, list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0], 50)

    return vae,emb

def print_top_words(beta, feature_names, n_top_words=50):
    print ('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        print(" ".join([feature_names[j]
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
        print('**********')
    print ('---------------End of Topics------------------')

def write_topwords(filename, beta, feature_names, n_top_words=50):
    file_write = open(filename, 'w+')
    for i in range(len(beta)):
        file_write.write(" ".join([feature_names[j]
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
        file_write.write("\n")

    file_write.close()

def setup_prior(n_z=3):
    gamma = np.zeros((len(vocab),n_z))
    gamma_bin = np.zeros((batch_size, len(vocab),n_z))

    full_vocab = SW_helper.read_file_to_list(constants_AA.const.FILE_SEED_WORD_LOCATION)
    for k in range(len(full_vocab)):
        for idx in range(len(full_vocab[k])):
            ivocab = vocab[full_vocab[k][idx]]
            gamma[ivocab, k] = 1.0
            gamma_bin[:, ivocab, :] = 1.0

    return (gamma, gamma_bin)

def printGamma(model):
    gamma = model.gamma_test()
    all_vocab = []

    list_seed_words = SW_helper.read_file_to_list(constants_AA.const.FILE_SEED_WORD_LOCATION)
    for a_list_seed_words in list_seed_words:
        all_vocab.extend(a_list_seed_words)

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    for idx in range(len(all_vocab)):
        ivocab = vocab[all_vocab[idx]]
        print (ivocab, all_vocab[idx], gamma[ivocab])

def calcPerp(model):
    cost=[]
    np.set_printoptions(precision=10)
    np.set_printoptions(suppress=False)

    batches_te = n_samples_te // batch_size
    for index_batch_te in range(batches_te):
        batch_data_te = data_te[index_batch_te*batch_size:(index_batch_te+1)*batch_size]
        batch_data_te = np.array([onehot(doc,vocab_size) for doc in batch_data_te])

        n_d = np.sum(batch_data_te)
        c=model.test(batch_data_te)
        cost.append(c/n_d)

    print ('The approximated perplexity is: ',(np.exp(np.mean(np.array(cost)))))

def main(argv):
    m = ''
    f = ''
    s = ''
    t = ''
    b = ''
    r = ''
    e = ''
    try:
      opts, args = getopt.getopt(argv,"hpnm:f:s:t:b:r:,e:",["default=","model=","layer1=","layer2=","num_topics=","batch_size=","learning_rate=","training_epochs"])
    except getopt.GetoptError:
        print ('CUDA_VISIBLE_DEVICES=0 python run.py -m <model> -f <#units> -s <#units> -t <#topics> -b <batch_size> -r <learning_rate [0,1] -e <training_epochs>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('CUDA_VISIBLE_DEVICES=0 python run.py -m <model> -f <#units> -s <#units> -t <#topics> -b <batch_size> -r <learning_rate [0,1]> -e <training_epochs>')
            sys.exit()
        elif opt == '-p':
            print ('Running with the Default settings for prodLDA...')
            print ('CUDA_VISIBLE_DEVICES=0 python run.py -m prodlda -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 100')
            m='prodlda'
            f=100
            s=100
            t=50
            b=200
            r=0.002
            e=100
        elif opt == '-n':
            print ('Running with the Default settings for NVLDA...')
            print ('CUDA_VISIBLE_DEVICES=0 python run.py -m nvlda -f 100 -s 100 -t 50 -b 200 -r 0.005 -e 300')
            m='nvlda'
            f=100
            s=100
            t=50
            b=200
            r=0.01
            e=300
        elif opt == "-m":
            m=arg
        elif opt == "-f":
            f=int(arg)
        elif opt == "-s":
            s=int(arg)
        elif opt == "-t":
            t=int(arg)
        elif opt == "-b":
            b=int(arg)
        elif opt == "-r":
            r=float(arg)
        elif opt == "-e":
            e=int(arg)


    # Setup prior
    global batch_size, data_prior
    batch_size = b
    data_prior = setup_prior(t)

    # time start the experiment
    t_start = time()

    network_architecture,batch_size,learning_rate=make_network(f,s,t,b,r)
    print (network_architecture)
    print (opts)
    vae,emb = train(network_architecture, m, training_epochs=e,batch_size=batch_size,learning_rate=learning_rate)
    printGamma(vae)
    print_top_words(emb, list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0])
    calcPerp(vae)
    # time end the experiment
    t_end = time() - t_start
    print ("Experiment time: ", "{:.4f}".format(t_end))

if __name__ == "__main__":
   main(sys.argv[1:])
