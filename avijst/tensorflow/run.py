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
from data import IMDBDataset, DataLoader, Onehotify, YOnehotify

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
    labels = config_dataset['labels']
    num_classes = config_dataset['num_classes']

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

    (train_x, train_y), (test_x, test_y) = dataset
    id_vocab = utils.sort_values(vocab)
    vocab_size = len(vocab)
    tfms_x = [Onehotify(vocab_size)]
    tfms_y = [YOnehotify(num_classes)]

    print (vocab_size)
    print (train_x.shape, train_y.shape)
    print (test_x.shape, test_y.shape)

    # Split data
    sss = StratifiedShuffleSplit(n_splits=exp, test_size=ratio, random_state=0)
    # splitted_data = sss.split(dataset_x, dataset_y)

    train_ds, test_ds = IMDBDataset(train_x, train_y, tfms_x, tfms_y), \
        IMDBDataset(test_x, test_y, tfms_x, tfms_y)
    
    train_dl, test_dl = DataLoader(train_ds, bs, False), DataLoader(test_ds, bs, False)

    for batch_train_x,  batch_train_y in train_dl:
        print (batch_train_x.shape, batch_train_y.shape)

    exit()

    model = AVIAD(n_encoder_1, n_encoder_2, 
                    vocab_size, n_latent, 
                    gamma_prior=gamma_prior, ld=ld, al=al, lr=lr, dr=dr)

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
