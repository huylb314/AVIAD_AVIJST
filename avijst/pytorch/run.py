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
from data import IMDBDataset, Onehotify, YToOnehot, Tensorify, \
                 Floatify, Padify, Lengthen, Cpuify, ToInt64, \
                 Numpyify, Longify, CheckAndCudify, Sampler, DataLoader, collate_imdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/1k.yaml",
                        help="Which configuration to use. See into 'config' folder")

    opt = parser.parse_args()
    opt_config = 'configs/imdb.yaml'
    # with open(opt.config, 'r') as ymlfile:
    with open(opt_config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print(config)
    
    # dataset
    config_dataset = config['dataset']
    folder_path = config_dataset['folder_path']
    data_path = osp.join(folder_path, config_dataset['data_file'])
    vocab_path = osp.join(folder_path, config_dataset['vocab_file'])
    labels = config_dataset['labels']
    maxlen = config_dataset['maxlen']
    n_classes = len(labels)

    # model
    config_model = config['model']
    n_encoder_1 = config_model['n_encoder_1']
    n_encoder_2 = config_model['n_encoder_2']
    n_latent = config_model['n_latent']
    n_sentiment = config_model['n_sentiment']
    dr = config_model['dropout']
    dr_senti = config_model['dropout_sentiment']
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

    tfms_xr = [Padify(maxlen), Numpyify(), Tensorify(), Longify(), CheckAndCudify()]
    tfms_lenxr = [Lengthen(), Numpyify(), Tensorify(), ToInt64()]

    tfms_x = [Numpyify(), Onehotify(vocab_size=vocab_size), Tensorify(), Floatify(), CheckAndCudify()]
    tfms_y = [Numpyify(), Tensorify(), CheckAndCudify()]

    criterion = nn.BCEWithLogitsLoss()
    model = AVIJST(vocab_size, n_encoder_1, n_encoder_2, n_latent, n_sentiment, dr, dr_senti, init_mult=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.99, 0.999))
    optimizer_pi = torch.optim.Adam(model.parameters(), cls_lr)
    if torch.cuda.is_available():
        model = model.cuda()
    # Split data
    sss = StratifiedShuffleSplit(n_splits=exp, test_size=n_labeled, random_state=0)
    splitted_train = sss.split(dataset_train_x, dataset_train_y)
    for _, ds_train_labeled_idx in splitted_train:
        train_unlabeled_ds = IMDBDataset(np.concatenate((dataset_train_x, dataset_test_x), axis=0),\
                                         np.concatenate((dataset_train_x, dataset_test_x), axis=0),\
                                         np.concatenate((dataset_train_x, dataset_test_x), axis=0),\
                                         np.concatenate((dataset_train_y, dataset_test_y), axis=0),\
                                         tfms_xr=tfms_xr, tfms_lenxr=tfms_lenxr, tfms_x=tfms_x, tfms_y=tfms_y)
        train_labeled_ds = IMDBDataset(dataset_train_x[ds_train_labeled_idx], dataset_train_x[ds_train_labeled_idx], \
                                       dataset_train_x[ds_train_labeled_idx], dataset_train_y[ds_train_labeled_idx], \
                                       tfms_xr=tfms_xr, tfms_lenxr=tfms_lenxr, tfms_x=tfms_x, tfms_y=tfms_y)
        test_ds = IMDBDataset(dataset_test_x, dataset_test_x, dataset_test_x, dataset_test_y, \
                              tfms_xr=tfms_xr, tfms_lenxr=tfms_lenxr, tfms_x=tfms_x, tfms_y=tfms_y)
        
        train_unlabeled_samp = Sampler(train_unlabeled_ds, bs, shuffle=False)
        train_labeled_samp = Sampler(train_labeled_ds, bs, shuffle=False)
        test_samp = Sampler(test_ds, bs, shuffle=False)

        train_unlabeled_dl = DataLoader(train_unlabeled_ds, sampler=train_unlabeled_samp, collate_fn=collate_imdb)
        train_labeled_dl = DataLoader(train_labeled_ds, sampler=train_labeled_samp, collate_fn=collate_imdb)
        test_dl = DataLoader(test_ds, sampler=test_samp, collate_fn=collate_imdb)

        start = time.time()
        print ("train_unlabeled_ds: ", len(train_unlabeled_ds))
        print ("train_labeled_ds: ", len(train_labeled_ds))
        print ("test_ds: ", len(test_ds))
        
        # train_labeled_iter = iter(train_labeled_dl)
        for epoch in range(epochs):
            model.train()
            avg_loss = 0.
            avg_cls_loss = 0.
            sum_t_c = 0.

            for idx, (train_raw_unlabeled_x, train_raw_len_unlabeled_x, train_unlabeled_x, train_unlabeled_y) in enumerate(train_unlabeled_dl):
                t_c = time.time()
                # Labeled
                loss_l = 0.0
                if len(train_labeled_ds) > 0:
                    for train_raw_labeled_x, train_raw_len_labeled_x, train_labeled_x, train_labeled_y in train_labeled_dl: 
                    # train_raw_labeled_x, train_raw_len_labeled_x, train_labeled_x, train_labeled_y = None, None, None, None
                    # try:
                    #     train_raw_labeled_x, train_raw_len_labeled_x, train_labeled_x, train_labeled_y = next(train_labeled_iter)
                    # except:
                    #     train_labeled_iter = iter(train_labeled_dl)
                    #     train_raw_labeled_x, train_raw_len_labeled_x, train_labeled_x, train_labeled_y= next(train_labeled_iter)
                    
                        recon, loss_l = model(train_raw_labeled_x, train_raw_len_labeled_x, train_labeled_x, train_labeled_y, cls_loss=True, compute_loss=True)
                        optimizer_pi.zero_grad()
                        loss_l.backward()
                        optimizer_pi.step()
                
                # Unlabeled
                recon, loss_u = model(train_raw_unlabeled_x, train_raw_len_unlabeled_x, train_unlabeled_x, train_unlabeled_y, cls_loss=False, compute_loss=True) 
                # optimize
                optimizer.zero_grad()        # clear previous gradients
                loss_u.backward()              # backprop
                optimizer.step()             # update parameters

                avg_loss += loss_u.item() / len(train_unlabeled_ds) * bs
                avg_cls_loss += loss_l.item() / len(train_unlabeled_ds) * bs
                c_elap = time.time() - t_c
                # Compute avg time
                sum_t_c += c_elap

            # Display logs per epoch step
            if (epoch + 1) % d_step == 0:
                model.eval()
                # weights = model.get_weights()
                print("##################################################")
                print("Epoch:", "%04d" % (epoch+1), \
                        "cost=", "{:.9f}".format(avg_loss), \
                        "cls_cost=", "{:.9f}".format(avg_cls_loss), \
                        "sum_t_c={:.2f}".format(sum_t_c))
                utils.classification_evaluate_dl(model, test_dl, n_sentiment, labels=['negative', 'positive'])
                # utils.print_top_words(epoch + 1, weights, id_vocab, n_topwords, result, write, printout=False)
                print("##################################################")

        # weights = model.get_weights()
        # utils.classification_evaluate_dl(model, train_dl, n_latent, labels, show=True)
        # utils.classification_evaluate_dl(model, test_dl, n_latent, labels, show=True)
        # utils.print_top_words(epoch + 1, weights, id_vocab, n_topwords, result, write)
        # utils.calc_perp(model, test_dl, gamma_prior_batch)

if __name__ == "__main__":
    main()
