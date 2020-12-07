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
parser.add_argument("--oc_folder", help="folder that contains oc files")
parser.add_argument("--number_topic", help="number of topics")
parser.add_argument("--final_log", help="log output for epoch", default="final_log.txt")


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
oc_files = []
wc_files = []
number_topic = args.number_topic
ref_corpus_dir = args.ref_corpus_dir
metric = args.metric
topns = args.topns
final_log = args.final_log

def parse_epoch_num(filename):
    num, ext = filename.split('.txt')
    return int(num)

#get the partitions of topic folder
for f in sorted(os.listdir(args.topic_folder), key=parse_epoch_num):
    if not f.startswith(".") and os.path.isfile(os.path.join(args.topic_folder, f)):
        topic_files.append(os.path.join(args.topic_folder, f))
        oc_files.append(os.path.join(args.oc_folder, f))
        wc_files.append(os.path.join(args.wc_folder, f))

import threading
import time
from multiprocessing import Pool

# locks
wc_lock = threading.Lock()

##################
#worker functions#
##################
def convert_to_index(wordlist, unigram_rev):
    ids = []

    for word in wordlist.split():
        if word in unigram_rev:
            ids.append(unigram_rev[word])
        else:
            ids.append(0)

    return ids

# update the word count of a given word


def update_word_count(word, worker_wordcount):
    count = 0
    if word in worker_wordcount:
        count = worker_wordcount[word]
    count += 1
    worker_wordcount[word] = count

    if debug:
        print("\tupdating word count for =", word)

# update the word count given a pair of words


def update_pair_word_count(w1, w2, topic_word_rel, worker_wordcount):
    if (w1 in topic_word_rel and w2 in topic_word_rel[w1]) or \
            (w2 in topic_word_rel and w1 in topic_word_rel[w2]):
        if w1 > w2:
            combined = w2 + "|" + w1
        else:
            combined = w1 + "|" + w2
        update_word_count(combined, worker_wordcount)

# given a sentence, find all ngrams (unigram or above)
def get_ngrams(words, topic_word_rel):
    if debug:
        for word in words:
            if word > 0:
                print (word, "=", unigram_list[word-1])

    all_ngrams = []
    ngram = []
    for i in range(0, len(words)):
        if (words[i] == 0):
            if len(ngram) > 0:
                all_ngrams.append(ngram)
                ngram = []
        else:
            ngram.append(unigram_list[words[i]-1])
    # append the last ngram
    if len(ngram) > 0:
        all_ngrams.append(ngram)
        ngram = []

    # permutation within ngrams
    ngrams_perm = []
    for ngram in all_ngrams:
        for i in range(1, len(ngram)+1):
            for j in range(0, len(ngram)-i+1):
                comb = [item for item in ngram[j:j+i]]
                ngrams_perm.append(' '.join(comb))

    # remove duplicates
    ngrams_perm = list(set(ngrams_perm))

    # only include ngrams that are found in topic words
    ngrams_final = []
    for ngram_perm in ngrams_perm:
        if ngram_perm in topic_word_rel:
            ngrams_final.append(ngram_perm)

    return ngrams_final

# calculate word counts, given a list of words


def calc_word_count(words, topic_word_rel, unigram_list, worker_wordcount):

    ngrams = get_ngrams(words, topic_word_rel)

    if debug:
        print("\nngrams =", ngrams, "\n")

    for ngram in ngrams:
        if (ngram in topic_word_rel):
            update_word_count(ngram, worker_wordcount)

    for w1_id in range(0, len(ngrams)-1):
        for w2_id in range(w1_id+1, len(ngrams)):
            if debug:
                print(
                    "\nChecking pair (", ngrams[w1_id], ",", ngrams[w2_id], ")")
            update_pair_word_count(
                ngrams[w1_id], ngrams[w2_id], topic_word_rel, worker_wordcount)

# primary worker function called by main


def calcwcngram(worker_num, window_size, corpus_file, topic_word_rel, unigram_list, unigram_rev):
    # now process the corpus file and sample the word counts
    line_num = 0
    worker_wordcount = {}
    total_windows = 0

    #sys.stderr.write("Worker " + str(worker_num) + " starts: " + str(time.time()) + "\n")
    for line in codecs.open(corpus_file, "r", "utf-8"):
        # convert the line into a list of word indexes
        words = convert_to_index(line, unigram_rev)

        if debug:
            print("====================================================================")
            print("line =", line)
            print("words =", " ".join([str(item) for item in words]))

        i = 0
        doc_len = len(words)
        # number of windows
        if window_size != 0:
            num_windows = doc_len + window_size - 1
        else:
            num_windows = 1
        # update the global total number of windows
        total_windows += num_windows

        for tail_id in range(1, num_windows+1):
            if window_size != 0:
                head_id = tail_id - window_size
                if head_id < 0:
                    head_id = 0
                words_in_window = words[head_id:tail_id]
            else:
                words_in_window = words

            if debug:
                print ("=========================")
                print ("line_num =", line_num)
                print ("words_in_window =", " ".join([str(item) for item in words_in_window]))

            calc_word_count(words_in_window, topic_word_rel, unigram_list,
                            worker_wordcount)

            i += 1

        line_num += 1

    # update the total windows seen for the worker
    worker_wordcount["!!<TOTAL_WINDOWS>!!"] = total_windows

    return worker_wordcount

################
#main functions#
################
# update the topic word - candidate words relation dictionary


def update_topic_word_rel(w1, w2):
    related_word_set = set([])
    if w1 in topic_word_rel:
        related_word_set = topic_word_rel[w1]
    if w2 != w1:
        related_word_set.add(w2)

    topic_word_rel[w1] = related_word_set

#########################################
#main functions ComputeObservedCoherence#
#########################################

import argparse
import sys
import operator
import math
import codecs
import numpy as np
from collections import defaultdict

###########
#functions#
###########
#use utf-8 for stdout
# sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

#compute the association between two words
def calc_assoc(word1, word2, metric):
    combined1 = word1 + "|" + word2
    combined2 = word2 + "|" + word1

    combined_count = 0
    if combined1 in wordcount:
        combined_count = wordcount[combined1]
    elif combined2 in wordcount:
        combined_count = wordcount[combined2]
    w1_count = 0
    if word1 in wordcount:
        w1_count = wordcount[word1]
    w2_count = 0
    if word2 in wordcount:
        w2_count = wordcount[word2]

    if (metric == "pmi") or (metric == "npmi"):
        if w1_count == 0 or w2_count == 0 or combined_count == 0:
            result = 0.0
        else:
            result = math.log((float(combined_count)*float(window_total))/ \
                float(w1_count*w2_count), 10)
            if metric == "npmi":
                result = result / (-1.0*math.log(float(combined_count)/(window_total),10))

    elif metric == "lcp":
        if combined_count == 0:
            if w2_count != 0:
                result = math.log(float(w2_count)/window_total, 10)
            else:
                result = math.log(float(1.0)/window_total, 10)
        else:
            result = math.log((float(combined_count))/(float(w1_count)), 10)

    return result

#compute topic coherence given a list of topic words
def calc_topic_coherence(topic_words, metric):
    topic_assoc = []
    for w1_id in range(0, len(topic_words)-1):
        target_word = topic_words[w1_id]
        #remove the underscore and sub it with space if it's a collocation/bigram
        w1 = " ".join(target_word.split(colloc_sep))
        for w2_id in range(w1_id+1, len(topic_words)):
            topic_word = topic_words[w2_id]
            #remove the underscore and sub it with space if it's a collocation/bigram
            w2 = " ".join(topic_word.split(colloc_sep))
            if target_word != topic_word:
                topic_assoc.append(calc_assoc(w1, w2, metric))

    return float(sum(topic_assoc))/len(topic_assoc)

######
#main#
######
if __name__ == '__main__':
    #process the word count file(s)
    for ctopic_file, cwc_file, coc_file in zip(topic_files, wc_files, oc_files):
        print (ctopic_file, cwc_file, coc_file)
        # parser arguments
        # desc = "Computes the word pair co-occurrences for topics. Parallel processing is achieved by \
        #     splitting the corpus into multiple partitions."
        # parser = argparse.ArgumentParser(description=desc)
        #####################
        #positional argument#
        #####################
        # parser.add_argument("topic_file", help="file that contains the topics")
        # parser.add_argument(
        #     "ref_corpus_dir", help="directory that contains the reference corpus")
        # args = parser.parse_args()

        ####################
        #call back function#
        ####################
        def calcwcngram_complete(worker_wordcount):
            wc_lock.acquire()
            global num_comp
            global ord_count

            # update the wordcount from the worker
            for k, v in worker_wordcount.items():
                curr_v = 0
                if k in word_count:
                    curr_v = word_count[k]
                curr_v += v
                word_count[k] = curr_v

            wc_lock.release()

        # parameters
        window_size = 20  # size of the sliding window; 0 = use document as window
        colloc_sep = "_"  # symbol for concatenating collocations
        debug = False

        # constants
        # key name for total number of windows (in wordcount)
        TOTALWKEY = "!!<TOTAL_WINDOWS>!!"

        #global variables
        # a dictionary that stores related topic words, e.g. { "space": set(["space", "earth", ...]), ... }
        topic_word_rel = {}
        unigram_list = []  # a list of unigrams (from topic words and candidates)
        unigram_rev = {}  # a reverse index of unigrams
        word_count = {}  # word counts (both single and pair)
        corpus_partitions = []  # a list of the partitions of the corpus

        # input
        topic_file = codecs.open(ctopic_file, "r", "utf-8")
        
        # get the partitions of the reference corpus
        for f in sorted(os.listdir(ref_corpus_dir)):
            if not f.startswith("."):
                corpus_partitions.append(ref_corpus_dir + "/" + f)

        # process the topic file and get the topic word relation
        unigram_set = set([])  # a set of all unigrams from the topic words
        for line in topic_file:
            line = line.strip()
            topic_words = line.split()

            # update the unigram list and topic word relation
            for word1 in topic_words:
                # update the unigram first
                for word in word1.split(colloc_sep):
                    unigram_set.add(word)

                # update the topic word relation
                for word2 in topic_words:
                    if word1 != word2:
                        # if it's collocation clean it so it's separated by spaces
                        cleaned_word1 = " ".join(word1.split(colloc_sep))
                        cleaned_word2 = " ".join(word2.split(colloc_sep))
                        update_topic_word_rel(cleaned_word1, cleaned_word2)
        # sort the unigrams and create a list and a reverse index
        unigram_list = sorted(list(unigram_set))
        unigram_rev = {}
        unigram_id = 1
        for unigram in unigram_list:
            unigram_rev[unigram] = unigram_id
            unigram_id += 1

        # spawn multiple threads to process the corpus
        po = Pool()
        for i, cp in enumerate(corpus_partitions):
            sys.stderr.write("creating a thread for corpus partition " + cp + "\n")
            sys.stderr.flush()
            po.apply_async(calcwcngram, (i, window_size, cp, topic_word_rel, unigram_list, unigram_rev,),
                        callback=calcwcngram_complete)
        po.close()
        po.join()

        with open(cwc_file, 'w+') as f:
            # all done, print the word counts
            for tuple in sorted(word_count.items()):
                f.write (tuple[0] + "|" + str(tuple[1]) + "\n")

        #parser arguments
        # desc = "Computes the observed coherence for a given topic and word-count file."
        # parser = argparse.ArgumentParser(description=desc)
        #####################
        #positional argument#
        #####################
        # parser.add_argument("topic_file", help="file that contains the topics")
        # parser.add_argument("metric", help="type of evaluation metric", choices=["pmi","npmi","lcp"])
        # parser.add_argument("wordcount_file", help="file that contains the word counts")

        ###################
        #optional argument#
        ###################
        # parser.add_argument("-t", "--topns", nargs="+", type=int, default=[10], \
        #     help="list of top-N topic words to consider for computing coherence; e.g. '-t 5 10' means it " + \
        #     " will compute coherence over top-5 words and top-10 words and then take the mean of both values." + \
        #     " Default = [10]")
        #constants
        WTOTALKEY = "!!<TOTAL_WINDOWS>!!" #key name for total number of windows (in word count file)

        #global variables
        window_total = 0 #total number of windows
        wordcount = {} #a dictionary of word counts, for single and pair words
        wordpos = {} #a dictionary of pos distribution

        #parameters
        colloc_sep = "_" #symbol for concatenating collocations
        #input
        topic_file = codecs.open(ctopic_file, "r", "utf-8")
        wc_file = codecs.open(cwc_file, "r", "utf-8")

        with open(coc_file, 'w+') as f:
            #process the word count file(s)
            for line in wc_file:
                line = line.strip()
                data = line.split("|")
                if len(data) == 2:
                    wordcount[data[0]] = int(data[1])
                elif len(data) == 3:
                    if data[0] < data[1]:
                        key = data[0] + "|" + data[1]
                    else:
                        key = data[1] + "|" + data[0]
                    wordcount[key] = int(data[2])
                else:
                    print ("ERROR: wordcount format incorrect. Line =", line)
                    raise SystemExit

            #get the total number of windows
            if WTOTALKEY in wordcount:
                window_total = wordcount[WTOTALKEY]

            #read the topic file and compute the observed coherence
            topic_coherence = defaultdict(list) # {topicid: [tc]}
            topic_tw = {} #{topicid: topN_topicwords}
            for topic_id, line in enumerate(topic_file):
                topic_list = line.split()[:max(topns)]
                topic_tw[topic_id] = " ".join(topic_list)
                for n in topns:
                    topic_coherence[topic_id].append(calc_topic_coherence(topic_list[:n], metric))

            #sort the topic coherence scores in terms of topic id
            tc_items = sorted(topic_coherence.items())
            mean_coherence_list = []
            for item in tc_items:
                topic_words = topic_tw[item[0]].split()
                mean_coherence = np.mean(item[1])
                mean_coherence_list.append(mean_coherence)
                f.write ("[%.2f] (" % mean_coherence),
                for i in item[1]:
                    f.write ("%.2f;" % i),
                f.write ("){}\n".format(topic_tw[item[0]]))

            #print the overall topic coherence for all topics
            f.write ("==========================================================================\n")
            f.write ("Average Topic Coherence = %.3f\n" % np.mean(mean_coherence_list))
            f.write ("Median Topic Coherence = %.3f\n" % np.median(mean_coherence_list))

            with open(final_log, 'a+') as f_log:
                f_log.write("{} {} {}\n".format(ctopic_file, np.mean(mean_coherence_list), np.median(mean_coherence_list)))
