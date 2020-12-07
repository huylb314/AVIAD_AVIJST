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
from models import AVIAD
from data import URSADataset, DataLoader, Onehotify, YToOnehot

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
    data_path = osp.join(
        config_dataset['folder-path'], config_dataset['data-file'])
    vocab_path = osp.join(
        config_dataset['folder-path'], config_dataset['vocab-file'])
    seedword_path = osp.join(
        config_dataset['folder-path'], config_dataset['sw-file'])
    labels = config_dataset['labels']

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
    bs = config_training['bs']
    d_step = config_training['d_step']
    epochs = config_training['epochs']
    n_topwords = config_training['n_topwords']
    ratio = config_training['ratio']
    exp = config_training['exp']
    result = config_training['result']
    write = config_training['write']

    # create result folders
    os.makedirs(result, exist_ok=True)

    dataset = np.load(data_path)
    with open(vocab_path, 'rb') as vocabfile:
        vocab = pickle.load(vocabfile)

    dataset_x, dataset_y = dataset[:, 0], dataset[:, 1]
    id_vocab = utils.sort_values(vocab)
    vocab_size = len(vocab)
    seedwords = utils.read_seedword(seedword_path)

    tfms_x = [Onehotify(vocab_size)]
    tfms_y = []

    # Split data
    sss = StratifiedShuffleSplit(n_splits=exp, test_size=ratio, random_state=0)
    splitted_data = sss.split(dataset_x, dataset_y)

    gamma_prior = np.zeros((vocab_size, n_latent))
    gamma_prior_batch = np.zeros((bs, vocab_size, n_latent))
    for idx_topic, seed_topic in enumerate(seedwords):
        for idx_word, seed_word in enumerate(seed_topic):
            idx_vocab = vocab[seed_word]
            gamma_prior[idx_vocab, idx_topic] = 1.0  # V x K
            gamma_prior_batch[:, idx_vocab, :] = 1.0  # N x V x K

    model = AVIAD(n_encoder_1, n_encoder_2, vocab_size, n_latent, 
                    gamma_prior=gamma_prior, ld=ld, al=al, lr=lr, dr=dr)

n_hidden_enc_1, n_hidden_enc_2, n_vocab, n_topic, gamma_prior, init_mult=1.0, variance=0.995, \
                        ld=20.0, al=1.0, lr=0.001, dr=0.6

    exit()

    for ds_train_idx, ds_test_idx in splitted_data:
        train_ds, test_ds = URSADataset(dataset_x[ds_train_idx], dataset_y[ds_train_idx], tfms_x, tfms_y), \
            URSADataset(dataset_x[ds_test_idx], dataset_y[ds_test_idx], tfms_x, tfms_y)
        
        train_dl, test_dl = DataLoader(train_ds, bs, False), DataLoader(test_ds, bs, False)
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
                utils.print_gamma(model, vocab, seedwords)
                print("##################################################")

            print("epoch={}, cost={:.9f}, sum_t_c={:.2f}".format((epoch + 1), avg_cost, sum_t_c))

        utils.classification_evaluate_dl(model, train_dl, n_latent, labels, show=True)
        utils.classification_evaluate_dl(model, test_dl, n_latent, labels, show=True)
        utils.print_gamma(model, vocab, seedwords)
        utils.calc_perp(model, test_dl, gamma_prior_batch)        


if __name__ == "__main__":
    main()