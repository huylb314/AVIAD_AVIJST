# that's all -- now any client-code can
import const
import os

# Any constant variables are attribute ONCE
const.PRINT_STATUS = True

const.FOLDER_DATASET = './data/User Review Structure Analysis (URSA)/1k'

const.TRAIN_FILENAME_SAVE = 'train.txt.npy'
const.VOCAB_FILENAME_SAVE = 'vocab.pkl'
const.XML_FILENAME = 'Classified_Corpus.xml'
const.FILE_LOG_FOOD_NOT_PASS = 'food_not_pass.txt'
const.FILE_LOG_STAFF_NOT_PASS = 'staff_not_pass.txt'
const.FILE_LOG_AMBIENCE_NOT_PASS = 'ambience_not_pass.txt'
const.FILE_SEED_WORD = 'seed_words.txt'

const.TRAIN_FILE_LOCATION = os.path.join(const.FOLDER_DATASET, const.TRAIN_FILENAME_SAVE)
const.VOCAB_FILE_LOCATION = os.path.join(const.FOLDER_DATASET, const.VOCAB_FILENAME_SAVE)
const.CORPUS_XML_FILE_LOCATION = os.path.join(const.FOLDER_DATASET, const.XML_FILENAME)
const.FILE_LOG_FOOD_NOT_PASS_LOCATION = os.path.join(const.FOLDER_DATASET, const.FILE_LOG_FOOD_NOT_PASS)
const.FILE_LOG_STAFF_NOT_PASS_LOCATION = os.path.join(const.FOLDER_DATASET, const.FILE_LOG_STAFF_NOT_PASS)
const.FILE_LOG_AMBIENCE_NOT_PASS_LOCATION = os.path.join(const.FOLDER_DATASET, const.FILE_LOG_AMBIENCE_NOT_PASS)
const.FILE_SEED_WORD_LOCATION = os.path.join(const.FOLDER_DATASET, const.FILE_SEED_WORD)

const.TAG_XML_REVIEW = './/Review'
const.TAG_NAME_POLARITY_ALLOWED = ['Positive', 'Negative', 'Neutral']
const.TAG_NAME_FOOD = 'Food'
const.TAG_NAME_STAFF = 'Staff'
const.TAG_NAME_AMBIENCE = 'Ambience'

const.LABEL_REVIEW_FOOD = 0
const.LABEL_REVIEW_STAFF = 1
const.LABEL_REVIEW_AMBIENCE = 2
const.LIST_LABEL = [const.LABEL_REVIEW_FOOD, const.LABEL_REVIEW_STAFF, const.LABEL_REVIEW_AMBIENCE]

const.LENGTH_FOOD_ALLOWED = -1
const.LENGTH_STAFF_ALLOWED = -1
const.LENGTH_AMBIENCE_ALLOWED = -1
const.MIN_FREQ_ALLOWED = -1

const.SAMPLE_INDEX_FROM = 0
const.SAMPLE_INDEX_TO = 5
