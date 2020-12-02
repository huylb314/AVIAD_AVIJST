import yaml
import argparse
import logging
import pickle
import time
import math
import sys
import numpy as np
import os.path as osp
from sklearn.model_selection import StratifiedShuffleSplit

import utils
from models import ProdLDA
from data import URSADataset, DataLoader

def compose(x, funcs, *args, **kwargs):
    for f in listify(funcs): 
        x = f(x, **kwargs)
    return x

class Onehotify():
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    def __call__(self, item):
        return np.bincount(item, minlength=self.vocab_size)

class YToOnehot():
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def __call__(self, item):
        categorical = np.zeros((1, self.num_classes))
        categorical[0, item] = 1
        return categorical

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
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    splitted_data = sss.split(dataset_x, dataset_y)

    gamma_prior = np.zeros((vocab_size, n_latent))
    gamma_prior_batch = np.zeros((bs, vocab_size, n_latent))
    for idx_topic, seed_topic in enumerate(seedwords):
        for idx_word, seed_word in enumerate(seed_topic):
            idx_vocab = vocab[seed_word]
            gamma_prior[idx_vocab, idx_topic] = 1.0  # V x K
            gamma_prior_batch[:, idx_vocab, :] = 1.0  # N x V x K

    # define model
    model = ProdLDA(n_encoder_1, n_encoder_2, 
                    vocab_size, vocab_size, n_latent, 
                    data_prior=(gamma_prior, gamma_prior_batch), ld=ld, al=al, lr=lr, bs=bs, dr=dr)

    for ds_train_idx, ds_test_idx in splitted_data:
        train_ds, test_ds = URSADataset(dataset_x[ds_train_idx], dataset_y[ds_train_idx], tfms_x, tfms_y), \
            URSADataset(dataset_x[ds_test_idx], dataset_y[ds_test_idx], tfms_x, tfms_y)
        
        train_dl, test_dl = DataLoader(train_ds, bs), DataLoader(test_ds, bs)
        beta = None

        for epoch in range(epochs):
            avg_cost = 0.
            sum_t_c = 0.

            for batch_train_x,  batch_train_y in train_dl:
                t_c = time.time()
                cost, beta = model.partial_fit(batch_train_x)
                c_elap = time.time() - t_c

                # Compute average loss
                avg_cost += cost / len(train_ds) * bs

                # Compute avg time
                sum_t_c += c_elap

                if np.isnan(avg_cost):
                    print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    sys.exit()

            # Display logs per epoch step
            if epoch % d_step == 0:
                print("##################################################")
                print("Epoch:", '%04d' % (epoch+1),
                      "\n",
                      "cost=", "{:.9f}".format(avg_cost))
                utils.print_top_words(beta, id_vocab, n_topwords)
                print("##################################################")

            print("epoch={}, cost={:.9f}".format(epoch, avg_cost))

        utils.classification_evaluate_dl(model, train_dl, n_latent, labels, show=True)
        utils.classification_evaluate_dl(model, test_dl, n_latent, labels, show=True)


if __name__ == "__main__":
    main()
