import yaml
import argparse
import logging
import pickle
import time
import math
import sys
import numpy as np
import os
import os.path as osp
from sklearn.model_selection import StratifiedShuffleSplit

import utils
from models import AVIJST
from data import Dataset, DataLoader, Onehotify, Padify, YOnehotify

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import (Add, Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Activation, 
                          Embedding, Conv1D, GlobalMaxPooling1D, 
                          Dropout, Conv2D, Conv2DTranspose, MaxPooling2D)
from keras.layers.merge import concatenate
from keras.regularizers import l2
from keras.initializers import RandomUniform
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Model
from keras import metrics
from keras.utils import np_utils
from keras import backend as K
from keras_tqdm import TQDMNotebookCallback
from sklearn.metrics import accuracy_score

def comp_accuracy(vae):
    y_pred = np.zeros_like(y_test)
    batches = len(x_test) // batch_size
    for i in range(batches):
        idx_from = i * batch_size
        idx_to = (i+1) * batch_size
        y_pred[idx_from:idx_to] = np.argmax(vae.senti_prop(x_test_pi[idx_from:idx_to]), axis=-1)

    score = accuracy_score(y_test, y_pred)
    return score

def batch_sequences_to_matrix(tokenizer, X):
    out_X = tokenizer.sequences_to_matrix(X, mode='binary')
    return out_X

# def printTopWord(vae, file):
#     id_vocab = get_id_to_word(index_from=3)
#     n_top_words = 30
#     weights = vae.get_weights()
#     for weight in weights:
#         file.write('WEIGHT\n')
#         for i in range(len(weight)):
#             file.write("Topics " + str(i) + "\n")
#             file.write(" ".join([id_vocab[j]
#                 for j in weight[i].argsort()[:-n_top_words - 1:-1]]) + "\n")

def main():
    # Hyper Parameters
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default="configs/1k.yaml",
    #                     help="Which configuration to use. See into 'config' folder")

    # opt = parser.parse_args()

    config_path = "configs/5k/imdb.yaml"
    # with open(opt.config, 'r') as ymlfile:
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print(config)
    # dataset
    config_dataset = config['dataset']
    folder_path = config_dataset['folder_path']
    data_path = osp.join(folder_path, config_dataset['data_file'])
    vocab_path = osp.join(folder_path, config_dataset['vocab_file'])
    labels = config_dataset['labels']
    maxlen = config_dataset['maxlen']
    num_classes = len(labels)

    # model
    config_model = config['model']
    n_encoder_1 = config_model['n_encoder_1']
    n_encoder_2 = config_model['n_encoder_2']
    n_latent = config_model['n_latent']
    dr = config_model['dropout']
    ld = config_model['lambda']
    al = config_model['alpha']

    # training
    config_training = config['training']
    lr = config_training['lr']
    cls_lr = config_training['cls_lr']
    bs = config_training['bs']
    d_step = config_training['d_step']
    epochs = config_training['epochs']
    n_topwords = config_training['n_topwords']
    n_labeled = config_training['n_labeled']
    ratio = config_training['ratio']
    exp = config_training['exp']
    result = config_training['result']
    write = config_training['write']

    # create result folders
    os.makedirs(result, exist_ok=True)

    dataset = np.load(data_path)
    with open(vocab_path, 'rb') as vocabfile:
        vocab = pickle.load(vocabfile)

    (dataset_train_x, dataset_train_y), (dataset_test_x, dataset_test_y) = dataset
    id_vocab = utils.sort_values(vocab)
    vocab_size = len(vocab)
    tfms_unlabeled_x = [Onehotify(vocab_size)]
    tfms_labeled_x = [Padify(maxlen)]
    tfms_y = [YOnehotify(num_classes)]

    network_architecture = dict(n_hidden_recog_1=n_encoder_1, # 1st layer encoder neurons
                                n_hidden_recog_2=n_encoder_2, # 2nd layer encoder neurons
                                n_hidden_gener_1=vocab_size, # 1st layer decoder neurons
                                n_input=vocab_size, # MNIST data input (img shape: 28*28)
                                n_input_pi=maxlen,
                                n_z=n_latent,
                                n_p=num_classes)

    # model = AVIJST(network_architecture, learning_rate=lr, cls_learning_rate=cls_lr, batch_size=bs)

    model = AVIJST(network_architecture,
                              learning_rate=lr,
                              cls_learning_rate=cls_lr,
                              batch_size=bs)

    X_unlabeled = np.concatenate((dataset_train_x, dataset_test_x), axis=0)
    X_unlabeled_pi = np.concatenate((dataset_train_x, dataset_test_x), axis=0)
    sss = StratifiedShuffleSplit(n_splits=exp, test_size=n_labeled, random_state=0)
    splitted_train = sss.split(dataset_train_x, dataset_train_y)
    sample_size = 500
    tokenizer = Tokenizer(num_words=vocab_size)
    n_samples_tr = dataset_train_x.shape[0] + dataset_test_x.shape[0]
    batch_size = bs
    for _, ds_train_labeled_idx in splitted_train:
        train_unlabeled_ds = Dataset(np.concatenate((dataset_train_x, dataset_test_x), axis=0),\
                                        np.concatenate((dataset_train_y, dataset_test_y), axis=0),\
                                        tfms_unlabeled_x, tfms_y)
        train_unlabeled_pi_ds = Dataset(np.concatenate((dataset_train_x, dataset_test_x), axis=0),\
                                        np.concatenate((dataset_train_y, dataset_test_y), axis=0),\
                                        tfms_labeled_x, tfms_y)
        train_labeled_ds = Dataset(dataset_train_x[ds_train_labeled_idx], dataset_train_y[ds_train_labeled_idx], tfms_labeled_x, tfms_y)
        test_ds = Dataset(dataset_test_x, dataset_test_y, tfms_labeled_x, tfms_y)
        
        train_unlabeled_dl = DataLoader(train_unlabeled_ds, bs, False)
        train_unlabeled_pi_dl = DataLoader(train_unlabeled_pi_ds, bs, False)
        train_labeled_dl = DataLoader(train_labeled_ds, bs, False)
        test_dl = DataLoader(test_ds, bs, False)
        
        X_labeled_pi = dataset_train_x[ds_train_labeled_idx]
        X_labeled = X_labeled_pi
        y_labeled = dataset_train_y[ds_train_labeled_idx]

        folder="topwords_"+str(sample_size)
        os.makedirs(folder, exist_ok=True)
        start = time.time()
        history = []
        print("Label: ", X_labeled.shape, " Unlabeled: ", X_unlabeled.shape)
        
        # Init VAE
        tf.reset_default_graph()
        vae = model
        
        def create_minibatch(data_size):
            rng = np.random.RandomState(10)

            while True:
                # Return random data samples of a size 'minibatch_size' at each iteration
                ixs = rng.randint(data_size, size=batch_size)
                yield ixs
                
        unlabel_minibatches = create_minibatch(n_samples_tr)
        label_minibatches = create_minibatch(X_labeled.shape[0])
        
        train_labeled_iter = iter(train_labeled_dl)
        for epoch in range(epochs):
            batches = len(X_unlabeled) // batch_size
            #with tnrange(batches, leave=False) as pbar:
            
            avg_loss = 0.
            avg_kl_s_loss = 0.
            avg_kl_z_loss = 0.
            avg_cls_loss = 0.

            
            # for i in range(batches):
            for idx, ((train_unlabeled_x, _), (train_unlabeled_pi_x, _)) in enumerate(zip(train_unlabeled_dl, train_unlabeled_pi_dl)):
                # Labeled
                loss_l = 0
                if len(X_labeled) > 0:
                    try:
                        train_labeled_x, train_labeled_y = next(train_labeled_iter)
                        loss_l = vae.cls_fit(train_labeled_x, train_labeled_y)
                    except:
                        train_labeled_iter = iter(train_labeled_dl)
                        train_labeled_x, train_labeled_y = next(train_labeled_iter)
                        loss_l = vae.cls_fit(train_labeled_x, train_labeled_y)
                
                # Unlabeled
                index_range = unlabel_minibatches.__next__()
                loss_u, kl_s_loss, kl_z_loss, emb = vae.partial_fit(train_unlabeled_x, train_unlabeled_pi_x)
                avg_cls_loss += loss_l / n_samples_tr * batch_size
                avg_loss += loss_u / n_samples_tr * batch_size
                avg_kl_s_loss += kl_s_loss / n_samples_tr * batch_size
                avg_kl_z_loss += kl_z_loss / n_samples_tr * batch_size

            # avg_cls_loss = 0.
            # for train_labeled_x, train_labeled_y in train_labeled_dl:
            #     loss_l = vae.cls_fit(train_labeled_x, train_labeled_y)
            #     avg_cls_loss += loss_l / len(train_labeled_ds) * batch_size

            # Display logs per epoch step
            if epoch % 2 == 0:
                y_pred = []
                batches = len(dataset_test_x) // batch_size
                y_data = []
                for i in range(batches):
                    idx_from = i * batch_size
                    idx_to = (i+1) * batch_size
                    temp_X_labeled_pi = sequence.pad_sequences(dataset_test_x[idx_from:idx_to], maxlen=maxlen)
                    y_data.extend(dataset_test_y[idx_from:idx_to])
                    y_pred.extend(np.argmax(vae.senti_prop(temp_X_labeled_pi), axis=-1))
                    # print (y_pred[idx_from:idx_to])
                utils.classification_evaluate(y_pred, y_data, labels=['negative', 'positive'])
            print("Epoch:", '%04d' % (epoch+1), \
                "cls_cost=", "{:.9f}".format(avg_cls_loss), \
                "cost=", "{:.9f}".format(avg_loss), \
                "kl_s=", "{:.9f}".format(avg_kl_s_loss), \
                "kl_z=", "{:.9f}".format(avg_kl_z_loss), \
                # "accu=", "{:.1f}".format(100. * accu), \
                )
            
            # if epoch % 10 == 0:
            #     file = open(folder+"/epoch_"+str(epoch)+".txt","w") 
            #     printTopWord(vae, file)
            #     file.close()
    
        done = time.time()
        elapsed = done - start
        print("Elapsed: ", elapsed)

    exit()
    # Split data
    sss = StratifiedShuffleSplit(n_splits=exp, test_size=n_labeled, random_state=0)
    splitted_train = sss.split(dataset_train_x, dataset_train_y)
    for _, ds_train_labeled_idx in splitted_train:
        train_unlabeled_ds = Dataset(np.concatenate((dataset_train_x, dataset_test_x), axis=0),\
                                     np.concatenate((dataset_train_y, dataset_test_y), axis=0),\
                                     tfms_unlabeled_x, tfms_y)
        train_unlabeled_pi_ds = Dataset(np.concatenate((dataset_train_x, dataset_test_x), axis=0),\
                                        np.concatenate((dataset_train_y, dataset_test_y), axis=0),\
                                        tfms_labeled_x, tfms_y)
        train_labeled_ds = Dataset(dataset_train_x[ds_train_labeled_idx], dataset_train_y[ds_train_labeled_idx], tfms_labeled_x, tfms_y)
        test_ds = Dataset(dataset_test_x, dataset_test_y, tfms_labeled_x, tfms_y)
        
        train_unlabeled_dl = DataLoader(train_unlabeled_ds, bs, False)
        train_unlabeled_pi_dl = DataLoader(train_unlabeled_pi_ds, bs, False)
        train_labeled_dl = DataLoader(train_labeled_ds, bs, False)
        test_dl = DataLoader(test_ds, bs, False)

        for epoch in range(epochs):
            avg_cls_loss = 0.
            avg_loss = 0.
            avg_kl_s_loss = 0.
            avg_kl_z_loss = 0.
            for idx, ((train_unlabeled_x, _), (train_unlabeled_pi_x, _)) in enumerate(zip(train_unlabeled_dl, train_unlabeled_pi_dl)):
                # Labeled
                loss_l = 0
                train_labeled_x = None
                for train_labeled_x, train_labeled_y in train_labeled_dl:
                    loss_l = model.cls_fit(train_labeled_x, train_labeled_y)
                
                loss_u, kl_s_loss, kl_z_loss, emb = model.partial_fit(train_unlabeled_x, train_unlabeled_pi_x)
                loss = loss_l + loss_u
            
                avg_loss += loss_u / len(train_unlabeled_ds) * bs
                avg_cls_loss += loss_l / len(train_unlabeled_ds) * bs
                avg_kl_s_loss += kl_s_loss / len(train_unlabeled_ds) * bs
                avg_kl_z_loss += kl_z_loss / len(train_unlabeled_ds) * bs


            print("Epoch:", '%04d' % (epoch+1), \
                    "cls_cost=", "{:.9f}".format(avg_cls_loss), \
                    "cost=", "{:.9f}".format(avg_loss), \
                    "kl_s=", "{:.9f}".format(avg_kl_s_loss), \
                    "kl_z=", "{:.9f}".format(avg_kl_z_loss), \
                    )

            pi_pred = []
            pi_label = []
            for test_x,  test_y in test_dl:
                pi_out = model.senti_prop(test_x)
                pi_out = np.argmax(pi_out, axis=-1)
                test_y = np.argmax(test_y, axis=-1)
                pi_pred.extend(pi_out)
                pi_label.extend(test_y)
            utils.classification_evaluate(pi_pred, pi_label, labels=['negative', 'positive'])

    exit()

    for batch_train_x,  batch_train_y in train_dl:
        print (batch_train_x.shape, batch_train_y.shape)
        beta = None

        for epoch in range(epochs):
            avg_cost = 0.
            sum_t_c = 0.

            for batch_train_x,  batch_train_y in train_dl:
                t_c = time.time()
                cost, beta = model.partial_fit(batch_train_x, gamma_prior_batch)
                c_elap = time.time() - t_c

                # Compute average loss
                avg_cost += cost / len(train_ds) * bs

                # Compute avg time
                sum_t_c += c_elap

                if np.isnan(avg_cost):
                    print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    sys.exit()

            # Display logs per epoch step
            if (epoch + 1) % d_step == 0:
                print("##################################################")
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                utils.print_top_words(epoch + 1, beta, id_vocab, n_topwords, result, write)
                print("##################################################")

            print("epoch={}, cost={:.9f}, sum_t_c={:.2f}".format((epoch + 1), avg_cost, sum_t_c))

        gamma = model.gamma_test()
        utils.print_gamma(gamma, vocab, seedwords)
        utils.classification_evaluate_dl(model, train_dl, n_latent, labels, show=True)
        utils.classification_evaluate_dl(model, test_dl, n_latent, labels, show=True)
        utils.print_top_words(epoch + 1, beta, id_vocab, n_topwords, result, write)
        utils.calc_perp(model, test_dl, gamma_prior_batch)  


if __name__ == "__main__":
    main()
