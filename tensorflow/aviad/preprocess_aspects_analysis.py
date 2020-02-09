"""
CORPUS XML: http://spidr-ursa.rutgers.edu/

Restaurant (1 - n) Reviews
Review (1 - n) Aspects

Review: Document
Aspect: Sentence
"""
import xml.etree.ElementTree as ET
from xml import etree

import numpy as np
import pickle
import os

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize,sent_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn

import constants_aspects_analysis as constants
import preprocess_helper as helper

from data_logger import DataLogger
import re

data_logger_util = DataLogger()

MESSAGE="""
Make sure FOLDER_DATASET is the main folder dataset(where Classified_Corpus.xml is located), not the sub folder(1k,2k...).\n
Because we want to reduce the amount of data upload .
        """

print (MESSAGE)

def find_listxml(acorpus, aname_tag):
    return acorpus.findall(aname_tag)

def string_nested_xml(axml):
    return ' '.join([the_aiter for the_aiter in axml.itertext()])

def get_firstchild(axml):
    try:
        if len(axml.getchildren()) > 0:
            return axml.getchildren()[0].tag
        else:
            raise (Exception('ListIndex', 'aXmlElement input has no children.'))
    except Exception as e:
        print (str(e))

def xml_unique_valid(axml, alist_tag_allowed):
    return (len(axml.getchildren()) == 0) or (get_firstchild(axml) in alist_tag_allowed)

def get_listsentence_unique(alist_xml, alist_tag_allowed):
    the_listsentence = []
    for the_axml in alist_xml:
        if xml_unique_valid(the_axml, alist_tag_allowed):
            the_listsentence.append(string_nested_xml(the_axml))

    return the_listsentence

def xml_name_valid(axml, atag_name):
    return axml.tag == atag_name

def get_listxml_child(alist_xml, atag_name):
    the_listreturn = []
    for the_axml in alist_xml:
        for the_achild in the_axml:
            if xml_name_valid(the_achild, atag_name):
                the_listreturn.append(the_achild)

    return the_listreturn

def data_process():
# if __name__ == '__main__':
    helper.variable_information(constants.const, constants.const.PRINT_STATUS)

    corpus_tree = ET.parse(constants.const.CORPUS_XML_FILE_LOCATION)
    corpus = corpus_tree.getroot()

    listdocument = find_listxml(corpus, constants.const.TAG_XML_REVIEW)

    listxml_child_food = get_listxml_child(listdocument, \
                                          constants.const.TAG_NAME_FOOD)
    listxml_child_staff = get_listxml_child(listdocument, \
                                           constants.const.TAG_NAME_STAFF)
    listxml_child_ambience = get_listxml_child(listdocument, \
                                              constants.const.TAG_NAME_AMBIENCE)

    helper.print_length_variables([listxml_child_food, listxml_child_staff, listxml_child_ambience], \
                                   ['listxml_child_food', 'listxml_child_staff', 'listxml_child_ambience'], \
                                   constants.const.PRINT_STATUS)

    ##### FOOD ##### STAFF ##### AMBIENCE
    listxml_unique_food = get_listsentence_unique(listxml_child_food, \
                                                    constants.const.TAG_NAME_POLARITY_ALLOWED)
    listxml_unique_staff = get_listsentence_unique(listxml_child_staff, \
                                                  constants.const.TAG_NAME_POLARITY_ALLOWED)
    listxml_unique_ambience = get_listsentence_unique(listxml_child_ambience, \
                                                  constants.const.TAG_NAME_POLARITY_ALLOWED)

    helper.print_length_variables([listxml_unique_food, listxml_unique_staff, listxml_unique_ambience], \
                                  ['listxml_unique_food', 'listxml_unique_staff', 'listxml_unique_ambience'], \
                                  constants.const.PRINT_STATUS)

    listprocessed_food = helper.process_listtext(listxml_unique_food, \
                                                 constants.const.LENGTH_FOOD_ALLOWED, \
                                                 constants.const.FILE_LOG_FOOD_NOT_PASS_LOCATION)
    listprocessed_staff = helper.process_listtext(listxml_unique_staff,\
                                                  constants.const.LENGTH_STAFF_ALLOWED,\
                                                  constants.const.FILE_LOG_STAFF_NOT_PASS_LOCATION)
    listprocessed_ambience = helper.process_listtext(listxml_unique_ambience, \
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
