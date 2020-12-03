import xml.etree.ElementTree as ET

import yaml
import argparse
import numpy as np
import pickle
import os
import os.path as osp

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import sentiwordnet as swn
from nltk import FreqDist

st = PorterStemmer()
stopwords_eng = list(stopwords.words("english"))
stopwords_addition = ["'s","...","'ve","``","''","'m",'--',"'ll","'d"]
stopwords_eng = set(stopwords_eng + stopwords_addition)

def alphabet(atext):
    return re.sub("[^a-zA-Z]", " ", atext)

def processify(s):
    alphabeted_s = alphabet(s)
    tokenized_s = nltk.word_tokenize(alphabeted_s.lower())
    stemmed_s = [st.stem(w) for w in tokenized_s if w not in stopwords_eng]

    return (stemmed_s, len(stemmed_s))

def list_processify(sentences, len_allowed, logfile):
    ret = []
    with open(logfile, "w") as f:
        for idx, sentence in enumerate(sentences):
            processed_s, len_s = processify(sentence)
            if len_s > len_allowed:
                ret.append(processed_s)
            else:
                f.write( '## ' + str(idx) + ' ##\n')
                f.write( '##:' + str(sentence) + '\n')
                f.write( '##:' + str(processed_s) + '\n')
                f.write( '##:' + str(len_s) + '\n')
                f.write( '##-----------------------------\n')

    return ret

def labelify(x, y):
    return (x, y)

def list_labelify(datas, label):
    ret = []
    for data in datas:
        ret.append(labelify(data, label))
    return ret

def word_valid(w):
    return w not in [""," "]

def create_vocab(sentences, min_freq):
    words = []
    for sentence in sentences:
        for word in sentence:
            words.append(word)
    words_freq = FreqDist(words)
    vocab = []
    for word, freq in words_freq.items():
        if freq > min_freq:
            if word_valid(word):
                vocab.append(word)

    vocab_sorted = sorted(vocab)
    #Assign a number corresponding to each word. Makes counting easier.
    vocab_sorted_dict = dict(zip(vocab_sorted, range(len(vocab_sorted))))
    return vocab_sorted, vocab_sorted_dict

def onehotify(sentence, vocab, vocab_dict):
    ret = []
    for word in sentence:
        if word in vocab:
            ret.append(vocab_dict[word])
    return ret

def list_onehotify(data_labeled, id_vocab, vocab):
    ret = []
    for x_, y_ in data_labeled:
        onehoted_x = onehotify(x_, id_vocab, vocab)
        if onehoted_x != []:
            ret.append((onehoted_x, y_))
    return ret

def nested_text_xml(xml):
    return ' '.join([xml_text for xml_text in xml.itertext()])

def firstchild(axml):
    try:
        if len(axml.getchildren()) > 0:
            return axml.getchildren()[0].tag
        else:
            raise (Exception('ListIndex', 'aXmlElement input has no children.'))
    except Exception as e:
        print (str(e))

def xml_unique_valid(xml, tag_allowed):
    return (len(xml.getchildren()) == 0) or (firstchild(xml) in tag_allowed)

def xmls_unique(xmls, tag_allowed):
    ret = []
    for xml in xmls:
        if xml_unique_valid(xml, tag_allowed):
            ret.append(nested_text_xml(xml))
    return ret

def xml_name_valid(axml, atag_name):
    return axml.tag == atag_name

def xmls_child(xmlses, tagxml):
    ret = []
    for xmls in xmlses:
        for xml in xmls:
            if xml_name_valid(xml, tagxml):
                ret.append(xml)
    return ret

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/aviad/ursa/preprocessing.yaml",
                        help="Which configuration to use. See into 'config' folder")

    opt = parser.parse_args()

    with open(opt.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print(config)

    # dataset
    config_dataset = config['dataset']
    dataset_name = config_dataset['name']
    dataset_folder_path = config_dataset['folder-path']
    dataset_data_file = config_dataset['data-file']
    dataset_train_file = config_dataset['train-file']
    dataset_vocab_file = config_dataset['vocab-file']
    dataset_data_path = osp.join(dataset_folder_path, dataset_data_file)
    dataset_train_path = osp.join(dataset_folder_path, dataset_train_file)
    dataset_vocab_path = osp.join(dataset_folder_path, dataset_vocab_file)

    # xml
    config_tagxml = config['tagxml']
    tagxml_review = config_tagxml['review']
    tagxml_polarity = config_tagxml['polarity']
    tagxml_food = config_tagxml['food']
    tagxml_staff = config_tagxml['staff']
    tagxml_ambience = config_tagxml['ambience']

    # label
    config_label = config['label']
    label_food = config_label['food']
    label_staff = config_label['staff']
    label_ambience = config_label['ambience']

    # limit
    config_limit = config['limit']
    limit_food = config_limit['food']
    limit_staff = config_limit['staff']
    limit_ambience = config_limit['ambience']
    limit_min_freq = config_limit['min_freq']

    # log
    config_log = config['log']
    log_food = config_log['food']
    log_staff = config_log['staff']
    log_ambience = config_log['ambience']
    log_food_path = osp.join(dataset_folder_path, log_food)
    log_staff_path = osp.join(dataset_folder_path, log_staff)
    log_ambience_path = osp.join(dataset_folder_path, log_ambience)

    corpus_tree = ET.parse(dataset_data_path)
    corpus = corpus_tree.getroot()

    xmls = corpus.findall(tagxml_review)
    xmls_food = xmls_child(xmls, tagxml_food)
    xmls_staff = xmls_child(xmls, tagxml_staff)
    xmls_ambience = xmls_child(xmls, tagxml_ambience)

    print (len(xmls_food))
    print (len(xmls_staff))
    print (len(xmls_ambience))

    xmls_unique_food = xmls_unique(xmls_food, tagxml_polarity)
    xmls_unique_staff = xmls_unique(xmls_staff, tagxml_polarity)
    xmls_unique_ambience = xmls_unique(xmls_ambience, tagxml_polarity)

    print (len(xmls_unique_food))
    print (len(xmls_unique_staff))
    print (len(xmls_unique_ambience))

    processed_food = list_processify(xmls_unique_food, limit_food, log_food_path)
    processed_staff = list_processify(xmls_unique_staff, limit_staff, log_staff_path)
    processed_ambience = list_processify(xmls_unique_ambience, limit_ambience, log_ambience_path)

    print (len(processed_food))
    print (len(processed_staff))
    print (len(processed_ambience))

    labeled_food = list_labelify(processed_food, label_food)
    labeled_staff = list_labelify(processed_staff, label_staff)
    labeled_ambience = list_labelify(processed_ambience, label_ambience)

    print (len(labeled_food))
    print (len(labeled_staff))
    print (len(labeled_ambience))

    print (xmls_unique_food[0])
    print (xmls_unique_staff[0])
    print (xmls_unique_ambience[0])

    id_vocab, vocab = create_vocab(processed_food + \
                                   processed_staff + \
                                    processed_ambience, \
                                    limit_min_freq)

    onehot_data = list_onehotify(labeled_food + labeled_staff + labeled_ambience, \
                                id_vocab, vocab)

    # Save Into File
    np.save (dataset_train_path, onehot_data)
    with open(dataset_vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

if __name__ == "__main__":
    main()