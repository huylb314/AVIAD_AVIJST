import xml.etree.ElementTree as ET
from xml import etree

import yaml
import argparse
import numpy as np
import pickle
import os
import os.path as osp

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize,sent_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn
import re

import constants_aspects_analysis as constants
import preprocess_helper as helper

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

def xmls_child(alist_xml, atag_name):
    the_listreturn = []
    for the_axml in alist_xml:
        for the_achild in the_axml:
            if xml_name_valid(the_achild, atag_name):
                the_listreturn.append(the_achild)

    return the_listreturn

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
    dataset_data_path = osp.join(dataset_folder_path, dataset_data_file)

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

    corpus_tree = ET.parse(dataset_data_path)
    corpus = corpus_tree.getroot()

    xmls = corpus.findall(tagxml_review)

    xmls_food = xmls_child(xmls, tagxml_food)
    xmls_staff = xmls_child(xmls, tagxml_staff)
    xmls_ambience = xmls_child(xmls, tagxml_ambience)

    print (len(xmls_food))
    print (len(xmls_staff))
    print (len(xmls_ambience))

    xmls_unique_food = xmls_unique(xmls_food, tagxml_food)
    xmls_unique_staff = xmls_unique(xmls_staff, tagxml_staff)
    xmls_unique_ambience = xmls_unique(xmls_ambience, tagxml_ambience)

    print (len(xmls_unique_food))
    print (len(xmls_unique_staff))
    print (len(xmls_unique_ambience))

    exit()
    listprocessed_food = helper.process_listtext(xmls_unique_food, \
                                                 constants.const.LENGTH_FOOD_ALLOWED, \
                                                 constants.const.FILE_LOG_FOOD_NOT_PASS_LOCATION)
    listprocessed_staff = helper.process_listtext(xmls_unique_staff,\
                                                  constants.const.LENGTH_STAFF_ALLOWED,\
                                                  constants.const.FILE_LOG_STAFF_NOT_PASS_LOCATION)
    listprocessed_ambience = helper.process_listtext(xmls_unique_ambience, \
                                                     constants.const.LENGTH_AMBIENCE_ALLOWED, \
                                                     constants.const.FILE_LOG_AMBIENCE_NOT_PASS_LOCATION)

    helper.print_length_variables([listprocessed_food, listprocessed_staff, listprocessed_ambience], \
                                  ['listprocessed_food', 'listprocessed_staff', 'listprocessed_ambience'], \
                                  constants.const.PRINT_STATUS)

    listlabeled_food = helper.label_listdata(listprocessed_food, \
                                             constants.const.LABEL_REVIEW_FOOD)
    listlabeled_staff = helper.label_listdata(listprocessed_staff, \
                                              constants.const.LABEL_REVIEW_STAFF)
    listlabeled_ambience = helper.label_listdata(listprocessed_ambience, \
                                                 constants.const.LABEL_REVIEW_AMBIENCE)

    helper.print_length_variables([listlabeled_food, listlabeled_staff, listlabeled_ambience], \
                                  ['listlabeled_food', 'listlabeled_staff', 'listlabeled_ambience'], \
                                  constants.const.PRINT_STATUS)

    helper.print_listsample([listxml_unique_ambience, listprocessed_ambience, listlabeled_ambience], \
                            ['listxml_unique_ambience', 'listprocessed_ambience', 'listlabeled_ambience'], \
                            constants.const.SAMPLE_INDEX_FROM, constants.const.SAMPLE_INDEX_TO, constants.const.PRINT_STATUS)

    vocab, vocab_dict = helper.create_vocab_listsentence(listprocessed_food + listprocessed_staff + listprocessed_ambience, constants.const.MIN_FREQ_ALLOWED)

    list_onehot = helper.create_listonehot(listlabeled_food + listlabeled_staff + listlabeled_ambience, \
                                           vocab, vocab_dict)

    # Save Into File
    np.save (constants.const.TRAIN_FILE_LOCATION, list_onehot)
    data_logger_util.pickle(constants.const.VOCAB_FILE_LOCATION, vocab_dict)

if __name__ == "__main__":
    main()