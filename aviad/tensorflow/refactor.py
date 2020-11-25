import yaml
import argparse
import logging
import pickle
import numpy as np
import os.path as osp
from sklearn.model_selection import StratifiedShuffleSplit

import utils
from models import ProdLDA

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/1k.yaml", \
                        help="Which configuration to use. See into 'config' folder")
                        
    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print (config)
    # dataset
    config_dataset = config['dataset']
    data_path = osp.join(config_dataset['folder-path'], config_dataset['data-file'])
    vocab_path = osp.join(config_dataset['folder-path'], config_dataset['vocab-file'])
    seedword_path = osp.join(config_dataset['folder-path'], config_dataset['sw-file'])

    # model
    config_model = config['model']
    n_encoder_1 = config_model['n_encoder_1']
    n_encoder_2 = config_model['n_encoder_2']
    n_latent = config_model['n_latent']
    dr = config_model['dropout']

    # training
    config_training = config['training']
    lr = config_training['lr']
    bs = config_training['bs']
    d_step = config_training['d_step']
    
    loaded_data = np.load(data_path)
    data, label = loaded_data[:, 0], loaded_data[:, 1]
    with open(vocab_path, 'rb') as vocabfile:
        vocab = pickle.load(vocabfile)
    vocab_size=len(vocab)

    # Split data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    # train_index, test_index = sss.split(data, label)
    splitted_data = sss.split(data, label)

    gamma = np.zeros((vocab_size, n_latent))
    gamma_bin = np.zeros((bs, vocab_size, n_latent))
    seedwords = utils.read_seedword(seedword_path)
    print (seedwords)
    for kidx, kseedword in enumerate(seedwords):
        for iidx, iseedword in enumerate(kseedword):
            ivocab = vocab[iseedword]
            gamma[ivocab, kidx] = 1.0
            gamma_bin[:, ivocab, :] = 1.0

    print (gamma_bin.shape)

    # define model
    model = ProdLDA(n_encoder_1, n_encoder_2, vocab_size,
                   vocab_size, n_latent, learning_rate=lr,
                   batch_size=bs, data_prior=(gamma, gamma_bin))

    for train_index, test_index in splitted_data:
        data_train, label_train = data[train_index], label[train_index]
        data_test, label_test = data[test_index], label[test_index]
        print ('Data Loaded')
        print (data.shape)
        print ('Dim Training Data',data_train.shape, vocab_size)
        print ('Dim Test Data',data_test.shape, vocab_size)

        

if __name__ == "__main__":
    main()