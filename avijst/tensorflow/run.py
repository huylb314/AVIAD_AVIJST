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

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/1k.yaml",
                        help="Which configuration to use. See into 'config' folder")

    opt = parser.parse_args()

    with open(opt.config, 'r') as ymlfile:
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

    model = AVIJST(network_architecture, learning_rate=lr, cls_learning_rate=cls_lr, batch_size=bs)

    # Split data
    sss = StratifiedShuffleSplit(n_splits=exp, test_size=n_labeled, random_state=0)
    splitted_train = sss.split(dataset_train_x, dataset_train_y)
    for ds_train_unlabeled_idx, ds_train_labeled_idx in splitted_train:
        train_unlabeled_ds = Dataset(dataset_train_x[ds_train_unlabeled_idx], dataset_train_y[ds_train_unlabeled_idx], tfms_unlabeled_x, tfms_y)
        train_labeled_ds = Dataset(dataset_train_x[ds_train_labeled_idx], dataset_train_y[ds_train_labeled_idx], tfms_labeled_x, tfms_y)
        test_ds = Dataset(dataset_test_x, dataset_test_y, tfms_labeled_x, tfms_y)
        
        train_unlabeled_dl, train_labeled_dl = DataLoader(train_unlabeled_ds, bs, False), DataLoader(train_labeled_ds, bs, False)
        test_dl = DataLoader(test_ds, bs, False)
        
        for epoch in range(epochs):
            avg_cls_loss = 0.
            avg_loss = 0.
            avg_kl_s_loss = 0.
            avg_kl_z_loss = 0.
            for train_unlabeled_x,  train_unlabeled_y in train_unlabeled_dl:
                # Labeled
                loss_l = 0
                train_labeled_x = None
                for train_labeled_x, train_labeled_y in train_labeled_dl:
                    loss_l = model.cls_fit(train_labeled_x, train_labeled_y)
                loss_u, kl_s_loss, kl_z_loss, emb = model.partial_fit(train_unlabeled_x, train_labeled_x)
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
                #   "accu=", "{:.1f}".format(100. * accu), \
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
