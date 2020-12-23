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
from data import IMDBDataset, Onehotify, YOnehotify, Tensorify, Floatify, Cudify
from torchvision.transforms import Compose

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
    n_classes = len(labels)

    # model
    config_model = config['model']
    n_encoder_1 = config_model['n_encoder_1']
    n_encoder_2 = config_model['n_encoder_2']
    n_latent = config_model['n_latent']
    n_sentiment = config_model['n_sentiment']
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

    tfms_xr = [Padify(maxlen), Numpyify(), Tensorify(), Longify(), CheckAndCudify()]
    tfms_lenxr = [Lengthen(), Numpyify(), Tensorify(), Longify(),  CheckAndCudify()]

    tfms_x = [Numpyify(), Onehotify(vocab_size=vocab_size), Tensorify(), Floatify(), CheckAndCudify()]
    tfms_y = [YToOnehot(num_classes=num_classes), Numpyify(), Tensorify(), Floatify(), CheckAndCudify()]

    model = AVIJST(vocab_size, n_encoder_1, n_encoder_2, \
                n_latent, n_classes, dr, \
                init_mult=1.0, num_layers=2, bidirectional=True, \
                pad_idx=id_vocab['<PAD>'])

    # Split data
    sss = StratifiedShuffleSplit(n_splits=exp, test_size=n_labeled, random_state=0)
    splitted_train = sss.split(dataset_train_x, dataset_train_y)
    for _, ds_train_labeled_idx in splitted_train:
        train_unlabeled_ds = IMDBDataset(np.concatenate((dataset_train_x, dataset_test_x), axis=0),\
                                         np.concatenate((dataset_train_y, dataset_test_y), axis=0),\
                                         np.concatenate((dataset_train_x, dataset_test_x), axis=0),\
                                         np.concatenate((dataset_train_y, dataset_test_y), axis=0),\
                                         tfms_xr=tfms_xr, tfms_lenxr=tfms_lenxr, tfms_x=tfms_x, tfms_y=tfms_y)
        train_labeled_ds = IMDBDataset(dataset_train_x[ds_train_labeled_idx], dataset_train_y[ds_train_labeled_idx], \
                                       dataset_train_x[ds_train_labeled_idx], dataset_train_y[ds_train_labeled_idx], \
                                       tfms_xr=tfms_xr, tfms_lenxr=tfms_lenxr, tfms_x=tfms_x, tfms_y=tfms_y)
        test_ds = IMDBDataset(dataset_test_x, dataset_test_y, dataset_test_x, dataset_test_y, \
                              tfms_xr=tfms_xr, tfms_lenxr=tfms_lenxr, tfms_x=tfms_x, tfms_y=tfms_y)
        
        train_unlabeled_samp = Sampler(train_unlabeled_ds, bs, shuffle=False)
        train_labeled_samp = Sampler(train_labeled_ds, bs, shuffle=False)
        test_samp = Sampler(test_ds, bs, shuffle=False)

        train_unlabeled_dl = DataLoader(train_unlabeled_ds, bs, drop_last=False)
        train_unlabeled_pi_dl = DataLoader(train_unlabeled_pi_ds, bs, drop_last=False)
        train_labeled_dl = DataLoader(train_labeled_ds, bs, drop_last=False)
        test_dl = DataLoader(test_ds, bs, drop_last=False)

        # CODE HERE

        start = time.time()
        print ("train_labeled_ds: ", len(train_labeled_ds))
        print ("train_unlabeled_pi_ds: ", len(train_unlabeled_pi_ds))
        print ("train_unlabeled_ds: ", len(train_unlabeled_ds))
        
        train_labeled_iter = iter(train_labeled_dl)
        for epoch in range(epochs):
            model.train()
            avg_loss = 0.
            avg_kl_s_loss = 0.
            avg_kl_z_loss = 0.
            avg_cls_loss = 0.
            sum_t_c = 0.

            for idx, ((train_unlabeled_x, _), (train_unlabeled_pi_x, _)) in enumerate(zip(train_unlabeled_dl, train_unlabeled_pi_dl)):
                t_c = time.time()
                # Labeled
                if len(train_labeled_ds) > 0:
                    train_labeled_x, train_labeled_y = None, None
                    try:
                        train_labeled_x, train_labeled_y = next(train_labeled_iter)
                    except:
                        train_labeled_iter = iter(train_labeled_dl)
                        train_labeled_x, train_labeled_y = next(train_labeled_iter)
                    
                    recon, theta_encoded, pi_encoded, \
                        theta_posterior_mean, theta_posterior_logvar, theta_posterior_var, \
                            pi_posterior_mean, pi_posterior_logvar, pi_posterior_var = \
                                model(x_semi, xr_semi, xrlen_semi, compute_loss=False)
                        loss_l = criterion(pi_posterior_mean, y_semi)
                    avg_cls_loss += loss_l / len(train_unlabeled_ds) * bs
                
                # Unlabeled
                loss_u, kl_s_loss, kl_z_loss, emb = model.partial_fit(train_unlabeled_x, train_unlabeled_pi_x)
                avg_loss += loss_u / len(train_unlabeled_ds) * bs
                avg_kl_s_loss += kl_s_loss / len(train_unlabeled_ds) * bs
                avg_kl_z_loss += kl_z_loss / len(train_unlabeled_ds) * bs
                c_elap = time.time() - t_c
                # Compute avg time
                sum_t_c += c_elap

            # Display logs per epoch step
            if (epoch + 1) % d_step == 0:
                weights = model.get_weights()
                print("##################################################")
                print("Epoch:", "%04d" % (epoch+1), \
                        "cls_cost=", "{:.9f}".format(avg_cls_loss), \
                        "cost=", "{:.9f}".format(avg_loss), \
                        "kl_s=", "{:.9f}".format(avg_kl_s_loss), \
                        "kl_z=", "{:.9f}".format(avg_kl_z_loss), \
                        "sum_t_c={:.2f}".format(sum_t_c))
                utils.classification_evaluate_dl(lambda x: model.senti_prop(x), test_dl, n_sentiment, labels=['negative', 'positive'])
                utils.print_top_words(epoch + 1, weights, id_vocab, n_topwords, result, write, printout=False)
                print("##################################################")

        weights = model.get_weights()
        utils.classification_evaluate_dl(model, train_dl, n_latent, labels, show=True)
        utils.classification_evaluate_dl(model, test_dl, n_latent, labels, show=True)
        utils.print_top_words(epoch + 1, weights, id_vocab, n_topwords, result, write)
        utils.calc_perp(model, test_dl, gamma_prior_batch)

if __name__ == "__main__":
    main()
