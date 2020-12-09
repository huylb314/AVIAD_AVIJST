import yaml
import argparse
import numpy as np
import pickle
import os
import os.path as osp

from keras.datasets import reuters, imdb

def create_vocab(idx_from=3):
    vocab = imdb.get_word_index()
    vocab = {k:(v + idx_from) for k,v in vocab.items()}
    vocab["<PAD>"] = 0
    vocab["<START>"] = 1
    vocab["<UNK>"] = 2
    vocab["<FCK>"] = 3
    
    id_vocab = {v:k for k,v in vocab.items()}
    return id_vocab, vocab

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/preprocessing.yaml",
                        help="Which configuration to use. See into 'config' folder")

    opt = parser.parse_args()

    with open(opt.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print(config)

    # dataset
    config_dataset = config['dataset']
    dataset_name = config_dataset['name']
    dataset_folder_path = config_dataset['folder-path']
    dataset_dest_path = config_dataset['dest-path']
    dataset_train_file = config_dataset['train-file']
    dataset_vocab_file = config_dataset['vocab-file']
    dataset_train_path = osp.join(dataset_dest_path, dataset_train_file)
    dataset_vocab_path = osp.join(dataset_dest_path, dataset_vocab_file)

    # label
    config_label = config['label']
    label_negative = config_label['negative']
    label_positive = config_label['positive']

    # limit
    config_limit = config['limit']
    limit_num_words = config_limit['num-words']
    limit_skip_top = config_limit['skip-top']
    limit_seed = config_limit['seed']
    limit_start_char = config_limit['start-char']
    limit_oov_char = config_limit['oov-char']
    limit_index_from = config_limit['index-from']
    limit_vocab_size = config_limit['vocab-size']

    # create result folders
    os.makedirs(dataset_dest_path, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=limit_num_words, skip_top=limit_skip_top, \
                                                          seed=limit_seed, start_char=limit_start_char, \
                                                          oov_char=limit_oov_char, index_from=limit_index_from)
    print (x_train.shape, y_train.shape)
    print (x_test.shape, y_test.shape)
    onehot_data = ((x_train, y_train), (x_test, y_test))
    id_vocab, vocab = create_vocab(idx_from=3)
    vocab = vocab.items()
    vocab = sorted(vocab, key=lambda item: item[1])
    vocab = dict(vocab[:limit_vocab_size])
    print (vocab)

    # Save Into File
    np.save (dataset_train_path, onehot_data)
    with open(dataset_vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

if __name__ == "__main__":
    main()