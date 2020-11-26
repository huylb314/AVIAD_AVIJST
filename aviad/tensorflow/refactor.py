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


class Dataset():
    def __init__(self, x, y): 
        self.x, self.y = x, y
    def __len__(self): 
        return len(self.x)
    def __getitem__(self, i): 
        return self.x[i], self.y[i]


class DataLoader():
    def __init__(self, ds, bs, drop_last=True): self.ds, self.bs, self.drop_last = ds, bs, drop_last
    def __iter__(self):
        length = len(self.ds) // self.bs if self.drop_last else math.ceil(len(self.ds) / self.bs)
        for i in range(0, length, 1):
            yield self.ds[(i*self.bs):(i*self.bs)+self.bs]


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
    epochs = config_training['epochs']

    dataset = np.load(data_path)
    with open(vocab_path, 'rb') as vocabfile:
        vocab = pickle.load(vocabfile)

    dataset_x, dataset_y = dataset[:, 0], dataset[:, 1]
    vocab_size = len(vocab)
    seedwords = utils.read_seedword(seedword_path)

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
    model = ProdLDA(n_encoder_1, n_encoder_2, vocab_size,
                    vocab_size, n_latent, learning_rate=lr,
                    batch_size=bs, data_prior=(gamma_prior, gamma_prior_batch))

    for ds_train_idx, ds_test_idx in splitted_data:
        train_ds, test_ds = Dataset(dataset_x[ds_train_idx], dataset_y[ds_train_idx]), \
            Dataset(dataset_x[ds_test_idx], dataset_y[ds_test_idx])
        train_dl, test_dl = DataLoader(train_ds, bs), DataLoader(test_ds, bs)

        for epoch in range(epochs):
            avg_cost = 0.
            avg_accuracy = 0.
            sum_t_c = 0.
            avg_precision = [0.] * 3
            avg_recall = [0.] * 3
            avg_f1_score = [0.] * 3

            for batch_train_x,  batch_train_y in train_dl:
                batch_train_x = np.array([utils.onehot(doc, vocab_size) for doc in batch_train_x])

                emb = None
                t_c = time.time()
                cost, emb = model.partial_fit(batch_train_x)
                c_elap = time.time() - t_c

                # Compute average loss
                avg_cost += cost / len(train_dl) * bs

                # Compute avg time
                sum_t_c += c_elap

                # Compute accuracy
                batch_train_theta = model.topic_prop(batch_train_x)
                batch_train_theta = np.argmax(batch_train_theta, axis=1)

                accuracy, precision, recall, f1_score = \
                    utils.classification_evaluate(batch_train_theta, batch_train_y,
                                                  ['food', 'staff', 'ambience'], show=False)
                avg_accuracy += accuracy / len(train_dl) * bs

                for k in range(3):
                    avg_precision[k] += precision[k] / len(train_dl) * bs
                    avg_recall[k] += recall[k] / len(train_dl) * bs
                    avg_f1_score[k] += f1_score[k] / len(train_dl) * bs

                if np.isnan(avg_cost):
                    print(
                        'Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    sys.exit()

            theta_y_pred_te = []
            batch_test_theta = []
            # test_theta = []
            # test_labe= = []
            for batch_index in test_dl:
                # print ("batch from {} to {}".format(batch_index, batch_index+bs if (batch_index+bs < len_train) else len_train))
                batch_x = data_test[batch_index:batch_index+bs]
                batch_x = np.array([utils.onehot(doc, vocab_size)
                                    for doc in batch_x])
                batch_y = label_test[batch_index:batch_index+bs]

                temp_theta_te = model.topic_prop(batch_x)
                temp_theta_max_idx_te = np.argmax(temp_theta_te, axis=1)
                # np.expand_dims(temp_theta_max_idx_te, axis=1)
                temp_theta_y_pred_te = temp_theta_max_idx_te

                theta_y_pred_te.extend(temp_theta_y_pred_te)
                theta_label_te.extend(batch_y)

            accuracy_te, precision_te, recall_te, f1_score_te = \
                utils.classification_evaluate(theta_y_pred_te, theta_label_te,
                                              ['food', 'staff', 'ambience'], show=False)

            # Display logs per epoch step
            if epoch % d_step == 0:
                print("##################################################",
                      "\n",
                      "Epoch:", '%04d' % (epoch+1),
                      "\n",
                      "cost=", "{:.9f}".format(avg_cost),
                      "avg_accuracy=", "{:.9f}".format(avg_accuracy),
                      "accuracy_te=", "{:.9f}".format(accuracy_te),
                      "total_calculate=", "{:.4f}".format(sum_t_c))

                for k in range(3):
                    print("avg_precision_{}".format(k), "=", "{:.9f}".format(avg_precision[k]),
                          "avg_recall_{}".format(
                              k), "=", "{:.9f}".format(avg_recall[k]),
                          "avg_f1_score_{}".format(k), "=", "{:.9f}".format(avg_f1_score[k]))
                    print("precision_te{}".format(k), "=", "{:.9f}".format(precision_te[k]),
                          "recall_te{}".format(
                              k), "=", "{:.9f}".format(recall_te[k]),
                          "f1_score_te{}".format(k), "=", "{:.9f}".format(f1_score_te[k]))
                # calcPerp(vae)
                print("##################################################")

            print("epoch={}, cost={:.9f}".format(epoch, avg_cost))


if __name__ == "__main__":
    main()
