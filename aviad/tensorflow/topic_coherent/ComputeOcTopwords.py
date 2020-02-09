import argparse
import sys
import operator
import math
import codecs
import numpy as np
from collections import defaultdict
import os


#parser arguments
desc = "Computes the observed coherence for a given topic and word-count file."
parser = argparse.ArgumentParser(description=desc)
#####################
#positional argument#
#####################
parser.add_argument("--metric", help="type of evaluation metric", choices=["pmi","npmi","lcp"], default="npmi")
parser.add_argument("--wc_folder", help="folder that contains the word counts", )
parser.add_argument("--ref_corpus_dir", help="folder that contains reference corpus")
parser.add_argument("--topic_folder", help="folder that contains topic files")
parser.add_argument("--topic_processed", help="folder that contains topic processed files")
parser.add_argument("--oc_folder", help="folder that contains oc files")
parser.add_argument("--number_topic", help="number of topics")
parser.add_argument("--number_label", help="number of labels")


###################
#optional argument#
###################
parser.add_argument("-t", "--topns", nargs="+", type=int, default=[10], \
    help="list of top-N topic words to consider for computing coherence; e.g. '-t 5 10' means it " + \
    " will compute coherence over top-5 words and top-10 words and then take the mean of both values." + \
    " Default = [10]")

args = parser.parse_args()

#parameters
colloc_sep = "_" #symbol for concatenating collocations

#global variables
#a dictionary that stores related topic words, e.g. { "space": set(["space", "earth", ...]), ... }
topic_files = [] #a list of the partitions of the corpus
processed_topic_files = []
oc_files = []
wc_files = []
number_topic = args.number_topic
number_label = args.number_label
ref_corpus_dir = args.ref_corpus_dir
metric = args.metric

#get the partitions of topic folder
for f in os.listdir(args.topic_folder):
    if not f.startswith(".") and os.path.isfile(args.topic_folder + "/" + f):
        topic_files.append(args.topic_folder + "/" + f)
        processed_topic_files.append(args.topic_processed + "/" + f)
        oc_files.append(args.oc_folder + "/" + f)
        wc_files.append(args.wc_folder + "/" + f)

###########
#functions#
###########
#use utf-8 for stdout
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

######
#main#
######

#process the word count file(s)
for ctopic_file, cprocessed_topic_file, cwc_file, coc_file in zip(topic_files, processed_topic_files, wc_files, oc_files):
    # python ComputeWordCount.py $topic_file $ref_corpus_dir > $wordcount_file
    print("Computing word occurrence...")
    print("python2 ComputeWordCount.py {0} {1} > {2}".format(ctopic_file, ref_corpus_dir, cwc_file))
    os.system("python2 ComputeWordCount.py {0} {1} > {2}".format(ctopic_file, ref_corpus_dir, cwc_file))

    # python ComputeObservedCoherence.py $topic_file $metric $wordcount_file > $oc_file
    print("Computing the observed coherence...")
    print("python2 ComputeObservedCoherence.py {0} {1} {2} > {3}".format(ctopic_file, metric, cwc_file, coc_file))
    os.system("python2 ComputeObservedCoherence.py {0} {1} {2} > {3}".format(ctopic_file, metric, cwc_file, coc_file))