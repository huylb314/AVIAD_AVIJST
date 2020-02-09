from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import sentiwordnet as swn
import nltk
from nltk import FreqDist

from data_logger import DataLogger

import re
from pprint import pprint

st = PorterStemmer()

def variable_information(avariable, aprint_status):
    if aprint_status == True:
        pprint(vars(avariable))

def alphabet(atext):
    return re.sub("[^a-zA-Z]", " ", atext)

def liststopword():
    the_listeng = list(stopwords.words("english"))
    the_listaddition = ["'s","...","'ve","``","''","'m",'--',"'ll","'d"]
    the_liststopword = set(the_listeng + the_listaddition)
    return the_liststopword

def process_atext(atext):
    the_alphabet = alphabet(atext)
    the_listword = nltk.word_tokenize(the_alphabet.lower())
    the_liststemmed = [st.stem(the_aword) for the_aword in the_listword if the_aword not in liststopword()]

    return (the_liststemmed, len(the_liststemmed))

def process_listtext(alist_text, alength_allowed, afile_log_notpass):
    the_file_log = open(afile_log_notpass, "w")
    the_listprocessed = []
    for the_index_text in range(len(alist_text)):
        the_atext = alist_text[the_index_text]
        the_aprocessed, the_alength = process_atext(the_atext)
        if the_alength > alength_allowed:
            the_listprocessed.append(the_aprocessed)
        else:
            the_file_log.write( '## ' + str(the_index_text) + ' ##\n')
            the_file_log.write( '##:' + str(the_atext) + '\n')
            the_file_log.write( '##:' + str(the_aprocessed) + '\n')
            the_file_log.write( '##:' + str(the_alength) + '\n')
            the_file_log.write( '##-----------------------------\n')

    the_file_log.close()
    return the_listprocessed

def label_adata(adata, alabel):
    return (adata, alabel)

def label_listdata(alist_data, alabel):
    the_listreturn = []
    for the_adata in alist_data:
        the_listreturn.append(label_adata(the_adata, alabel))
    return the_listreturn

def word_valid(aword):
    return aword not in [""," "]

def create_vocab_listsentence(alist_sentence, amin_freq_allowed):
    the_words = []
    for the_asentence in alist_sentence:
        for the_aword in the_asentence:
            the_words.append(the_aword)
    the_words_freq = FreqDist(the_words)
    the_vocab = []
    for the_aword, the_afreq in the_words_freq.items():
        if the_afreq > amin_freq_allowed:
            if word_valid(the_aword):
                the_vocab.append(the_aword)

    the_vocab_sorted = sorted(the_vocab)
    #Assign a number corresponding to each word. Makes counting easier.
    the_vocab_sorted_dict = dict(zip(the_vocab_sorted, range(len(the_vocab_sorted))))
    return the_vocab_sorted, the_vocab_sorted_dict

def onehot_valid(aonehot):
    return aonehot != []

def create_aonehot(adata_tokenized, avocab, avocab_dict):
    the_aonehot = []
    for the_aword in adata_tokenized:
        if the_aword in avocab:
            the_aonehot.append(avocab_dict[the_aword])

    return the_aonehot

def create_listonehot(alist_data_labeled, avocab, avocab_dict):
    the_listonehot = []
    for the_adata_tokenized, the_alabel in alist_data_labeled:
        the_aonehot = create_aonehot(the_adata_tokenized, avocab, avocab_dict)
        if onehot_valid(the_aonehot):
            the_listonehot.append((create_aonehot(the_adata_tokenized, avocab, avocab_dict), the_alabel))
    return the_listonehot

def print_length_avariable(avariable, astring_name, aprint_status):
    if aprint_status == True:
        print ('Length ', astring_name, \
              '\t: ', len(avariable))

def print_length_variables(alist_variable, alist_string_name, aprint_status):
    for the_avariable, the_astring_name in zip(alist_variable, alist_string_name):
        print_length_avariable(the_avariable, the_astring_name, aprint_status)

def print_asample(avariable, astring_name, aint_from, aint_to, aprint_status):
    if aprint_status == True:
        print ('### ', astring_name, '###')
        for the_asentence in avariable[aint_from:aint_to]:
            print ('#', the_asentence)

def print_listsample(alist_variable, alist_string_name, aint_from, aint_to, aprint_status):
    for the_avariable, the_astring_name in zip(alist_variable,alist_string_name):
        print_asample(the_avariable, the_astring_name, aint_from, aint_to, aprint_status)
