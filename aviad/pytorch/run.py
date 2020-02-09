import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

from collections import defaultdict
from itertools import groupby
from sklearn import datasets
from numpy import random
from scipy.stats import dirichlet, norm, poisson

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import os

from pathlib import Path
from collections import OrderedDict
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt

Path.ls = lambda x: list(x.iterdir())

folder_ds_path = Path('./data/User Review Structure Analysis (URSA)/')
xml_path = (folder_ds_path/'Classified_Corpus.xml')
ds_path = (folder_ds_path/'1k')
sentence_npy_path = (folder_ds_path/'sentence.npy')
vocab_pkl_path = (ds_path/'vocab.pkl')
seed_words_path = (ds_path/'seed_words.txt')
train_filename = (ds_path/'train.txt.npy')

# log words not pass
aspect_tags = ['Food', 'Staff', 'Ambience']
polatiry_tags = ['Positive', 'Negative', 'Neutral']
xml_review_tag = './/Review'
log_np = [[], [], []]

# length allowed sentences
# length_allowed = [11, 7, 4]
# min_freq_allowed = -1

vocab2id = pickle.load(open(vocab_pkl_path, 'rb'))
vocab_size=len(vocab2id)
train_data = np.load((train_filename), allow_pickle=True)
p_sentence_list, label_list = train_data[:, 0], train_data[:, 1]
vocab = dict(map(reversed, vocab2id.items()))
vocab_size = len(vocab)

from sklearn.model_selection import train_test_split

x_, y_ = [], []
for p_sentence, label_ in zip(p_sentence_list, label_list): 
    x_.append(p_sentence)
    y_.append(label_)

print (len(x_) == len(y_))

train_x, test_x, train_y, test_y =  train_test_split(
    x_, y_, test_size=0.1, random_state=0)

print ('Data Loaded')
print ('Dim Training Data',len(train_x), vocab_size)
print ('Dim Test Data', len(test_x), vocab_size)

bs = 200
en1_units=100
en2_units=100
num_topic=3
num_input=vocab_size
variance=0.995
init_mult=1.0
learning_rate=0.0005
batch_size=200
momentum=0.99
num_epoch=200
nogpu=True
drop_rate=0.6

def read_file_seed_words(fn):
    with open(fn, "r") as fr:
        def p_string_sw(l):
            return l.replace('\n','').split(',')
        rl = [p_string_sw(l) for l in fr]
    return rl

seed_words = read_file_seed_words(seed_words_path)
print (seed_words)

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]
def setify(o): return o if isinstance(o,set) else set(listify(o))
def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key): x = f(x, **kwargs)
    return x

def setup_prior(fn, n_k=3):
    gamma = torch.zeros((len(vocab),n_k))
    gamma_bin = torch.zeros((1, len(vocab),n_k))

    full_vocab = read_file_seed_words(fn)
    for k in range(len(full_vocab)):
        for idx in range(len(full_vocab[k])):
            ivocab = vocab2id[full_vocab[k][idx]]
            gamma[ivocab, k] = 1.0
            gamma_bin[:, ivocab, :] = 1.0

    return (gamma, gamma_bin)

def print_perp(model):
    cost = []
    model.eval()                        # switch to testing mode
    for x_test, y_test in test_dl:
        recon, loss = model(x_test, compute_loss=True, avg_loss=False)
        loss = loss.data
        counts = x_test.sum(1)
        cost.extend((loss / counts).data.cpu().tolist())
    print('The approximated perplexity is: ', (np.exp(np.mean(np.array(cost)))))

def print_top_words(beta, feature_names, n_top_words=10):
    print ('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        line = " ".join([feature_names[j] 
                         for j in beta[i].argsort()[:-n_top_words - 1:-1]])
        print('{}'.format(line))
    print ('---------------End of Topics------------------')
    
def print_gamma(gamma, seed_words, vocab, vocab2id):
    sws = []        
    for k in range(len(seed_words)):
        for idx in range(len(seed_words[k])):
            w = seed_words[k][idx]
            sws.append((k, w))

    for idx in range(len(sws)):
        k, w = sws[idx]
        ivocab = vocab2id[w]
        mk = gamma[ivocab].argmax(-1)
        print (ivocab, w, k, mk, gamma[ivocab])

def write_topwords(filename, beta, feature_names, n_top_words=50):
    file_write = open(filename, 'w+')
    for i in range(len(beta)):
        file_write.write(" ".join([feature_names[j]
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
        file_write.write("\n")

    file_write.close()

def collate(b):
    x, y = zip(*b)
    return torch.stack(x), torch.stack(y)

class IdifyAndLimitedVocab():
    _order=-1
    def __init__(self, vocab2id, limited_vocab):
        self.vocab2id = vocab2id
        self.limited_vocab = limited_vocab
    def __call__(self, item):
        idlist = [self.vocab2id[w] for w in item if self.vocab2id[w] < self.limited_vocab]
        return np.array(idlist)
    

class Numpyify():
    _order=0
    def __call__(self, item):
        return np.array(item)

class Onehotify():
    _order=1
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    def __call__(self, item):
        return np.array(np.bincount(item.astype('int'), minlength=self.vocab_size))
    
class YToOnehot():
    _order=1
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def __call__(self, item):
        categorical = np.zeros((1, self.num_classes))
        categorical[0, item] = 1
        return categorical

class Tensorify():
    _order=2
    def __call__(self, item):
        return torch.from_numpy(item)

class Floatify():
    _order=3
    def __call__(self, item):
        return item.float()
    
class CheckAndCudify():
    _order=100
    def __init__(self):
        self.ic = torch.cuda.is_available()
    def __call__(self, item):
        return item.cuda() if self.ic else item
    
class URSADataset(Dataset):
    def __init__(self, x, y, tfms_x, tfms_y): 
        self.x, self.y = x, y
        self.x_tfms = tfms_x
        self.y_tfms = tfms_y
    def __len__(self): 
        return len(self.x)
    def __getitem__(self, i): 
        return compose(self.x[i], self.x_tfms), compose(self.y[i], self.y_tfms)
    
class Sampler():
    def __init__(self, ds, bs, shuffle=False):
        self.n,self.bs,self.shuffle = len(ds),bs,shuffle
        
    def __iter__(self):
        self.idxs = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.bs): yield self.idxs[i:i+self.bs]

class DataLoader():
    def __init__(self, ds, sampler, collate_fn=collate):
        self.ds,self.sampler,self.collate_fn = ds,sampler,collate_fn
        
    def __iter__(self):
        for s in self.sampler: yield self.collate_fn([self.ds[i] for i in s])

num_classes = np.max(train_y) + 1
print("num_classes:", num_classes)

tfms_x = [Numpyify(), Onehotify(vocab_size=vocab_size), Tensorify(), Floatify(), CheckAndCudify()]
tfms_y = [YToOnehot(num_classes=num_classes), Tensorify(), Floatify(), CheckAndCudify()]

train_ds = URSADataset(train_x, train_y, tfms_x=tfms_x, tfms_y=tfms_y)
test_ds = URSADataset(test_x, test_y, tfms_x=tfms_x, tfms_y=tfms_y)

train_samp = Sampler(train_ds, bs, shuffle=False)
test_samp = Sampler(test_ds, bs, shuffle=False)

train_dl = DataLoader(train_ds, sampler=train_samp, collate_fn=collate)
test_dl = DataLoader(test_ds, sampler=test_samp, collate_fn=collate)

gamma_prior = setup_prior(seed_words_path, num_topic)
gamma, gamma_bin = gamma_prior

class ProdLDA(nn.Module):
    def __init__(self, num_input, en1_units, en2_units, num_topic, drop_rate, init_mult, gamma_prior):
        super(ProdLDA, self).__init__()
        self.num_input, self.en1_units, self.en2_units, \
        self.num_topic, self.drop_rate, self.init_mult = num_input, en1_units, en2_units, \
                                                            num_topic, drop_rate, init_mult
        # gamma prior
        self.gamma_prior = gamma_prior
        
        # encoder
        self.en1_fc = nn.Linear(num_input, en1_units)
        self.en1_ac = nn.Softplus()
        self.en2_fc     = nn.Linear(en1_units, en2_units)
        self.en2_ac = nn.Softplus()
        self.en2_dr   = nn.Dropout(drop_rate)
        
        # mean, logvar
        self.mean_fc = nn.Linear(en2_units, num_topic)
        self.mean_bn = nn.BatchNorm1d(num_topic)
        self.logvar_fc = nn.Linear(en2_units, num_topic)
        self.logvar_bn = nn.BatchNorm1d(num_topic)

        # decoder
        self.de_ac1 = nn.Softmax(dim=-1)
        self.de_dr = nn.Dropout(drop_rate)
        self.de_fc = nn.Linear(num_topic, num_input)
        self.de_bn = nn.BatchNorm1d(num_input)
        self.de_ac2 = nn.Softmax(dim=-1)
        
        # prior mean and variance as constant buffers
        self.prior_mean   = torch.Tensor(1, num_topic).fill_(0)
        self.prior_var    = torch.Tensor(1, num_topic).fill_(variance)
        self.prior_mean   = nn.Parameter(self.prior_mean, requires_grad=False)
        self.prior_var    = nn.Parameter(self.prior_var, requires_grad=False)
        self.prior_logvar = nn.Parameter(self.prior_var.log(), requires_grad=False)
        # initialize decoder weight
        if init_mult != 0:
            #std = 1. / math.sqrt( init_mult * (num_topic + num_input))
            self.de_fc.weight.data.uniform_(0, init_mult)
        # remove BN's scale parameters
        for component in [self.mean_bn, self.logvar_bn, self.de_bn]:
            component.weight.requires_grad = False
            component.weight.fill_(1.0)
        
    def gamma(self):
        # this function have to run after self.encode
        encoder_w1 = self.en1_fc.weight
        encoder_b1 = self.en1_fc.bias
        encoder_w2 = self.en2_fc.weight
        encoder_b2 = self.en2_fc.bias
        mean_w = self.mean_fc.weight
        mean_b = self.mean_fc.bias
        mean_running_mean = self.mean_bn.running_mean
        mean_running_var = self.mean_bn.running_var
        logvar_w = self.logvar_fc.weight
        logvar_b = self.logvar_fc.bias
        logvar_running_mean = self.logvar_bn.running_mean
        logvar_running_var = self.logvar_bn.running_var
        
        w1 = F.softplus(encoder_w1.t() + encoder_b1)
        w2 = F.softplus(F.linear(w1, encoder_w2, encoder_b2))
        wdr = F.dropout(w2, self.drop_rate)
        wo_mean = F.softmax(F.linear(wdr, mean_w, mean_b), dim=-1)
        wo_logvar = F.softmax(F.batch_norm(F.linear(wdr, logvar_w, logvar_b), logvar_running_mean, logvar_running_var), dim=-1)
        
        return wo_mean, wo_logvar
            
    def encode(self, input_):
        # encoder
        encoded1 = self.en1_fc(input_)
        encoded1_ac = self.en1_ac(encoded1)
        encoded2 = self.en2_fc(encoded1_ac)
        encoded2_ac = self.en2_ac(encoded2)
        encoded2_dr = self.en2_dr(encoded2_ac)
        
        encoded = encoded2_dr
        
        # hidden => mean, logvar
        mean_theta = self.mean_fc(encoded)
        mean_theta_bn = self.mean_bn(mean_theta)
        logvar_theta = self.logvar_fc(encoded)
        logvar_theta_bn = self.logvar_bn(logvar_theta)
        
        posterior_mean = mean_theta_bn
        posterior_logvar = logvar_theta_bn
        return encoded, posterior_mean, posterior_logvar
    
    def decode(self, input_, posterior_mean, posterior_var):
        # take sample
        eps = input_.data.new().resize_as_(posterior_mean.data).normal_() # noise 
        z = posterior_mean + posterior_var.sqrt() * eps                   # reparameterization
        # do reconstruction
        # decoder
        decoded1_ac = self.de_ac1(z)
        decoded1_dr = self.de_dr(decoded1_ac)
        decoded2 = self.de_fc(decoded1_dr)
        decoded2_bn = self.de_bn(decoded2)
        decoded2_ac = self.de_ac2(decoded2_bn)
        recon = decoded2_ac          # reconstructed distribution over vocabulary
        return recon
    
    def forward(self, input_, compute_loss=False, avg_loss=True):
        # compute posterior
        en2, posterior_mean, posterior_logvar = self.encode(input_) 
        posterior_var    = posterior_logvar.exp()
        
        recon = self.decode(input_, posterior_mean, posterior_var)
        if compute_loss:
            return recon, self.loss(input_, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)
        else:
            return recon

    def loss(self, input_, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL  = -(input_ * (recon + 1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017, 
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = self.prior_mean.expand_as(posterior_mean)
        prior_var    = self.prior_var.expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.num_topic)
        
        # gamma
        n, _ = input_.size()
        gamma_mean, gamma_logvar = self.gamma()
        gamma_prior, gammar_prior_bin = self.gamma_prior
        input_t = (input_ > 0).unsqueeze(dim=-1)
        input_bin = ((gammar_prior_bin.expand(n, -1, -1) == 1) & input_t)
        lambda_c = 20.0
        
        gamma_prior = gamma_prior.expand(n, -1, -1)      
        
        GL = lambda_c * ((gamma_prior - (input_bin.int()*gamma_mean))**2).sum((1, 2))
        
        # loss
        loss = (NL + KLD + GL)
        
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean()
        else:
            return loss

from sklearn import metrics

def compute_accuracy(y_pred, y_true):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_true=y_true, \
                                                     y_pred=y_pred, \
                                                     average=None)

    return (accuracy, precision, recall, f1_score)

model = ProdLDA(num_input, en1_units, en2_units, num_topic, drop_rate, init_mult, gamma_prior)
optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(momentum, 0.999))

if torch.cuda.is_available():
    model = model.cuda()

for epoch in range(num_epoch):
    loss_epoch = 0.0
    model.train()                    # switch to training mode
    for input_, label_ in train_dl:
        recon, loss = model(input_, compute_loss=True)
        # optimize
        optimizer.zero_grad()        # clear previous gradients
        loss.backward()              # backprop
        optimizer.step()             # update parameters
        # report
        loss_epoch += loss.item()    # add loss to loss_epoch
    if (epoch + 1) % 10 == 0:
        model.eval()
        # Test Model
        pred_train = []
        label_train = []
        pred_test = []
        label_test = []
        
        for x_train, y_train in train_dl:
            encoded, theta_mean, theta_logvar = model.encode(x_train)
            temp_theta_mean = theta_mean.argmax(-1).int().data.cpu().tolist()
            temp_y_train = y_train.argmax(-1).flatten().data.cpu().tolist()
            
            pred_train.extend(temp_theta_mean)
            label_train.extend(temp_y_train)
        
        accuracy_train, precision_train, recall_train, f1_score_train = compute_accuracy(pred_train, label_train)
        
        for x_test, y_test in test_dl:
            encoded, theta_mean, theta_logvar = model.encode(x_test)
            temp_theta_mean = theta_mean.argmax(-1).int().data.cpu().tolist()
            temp_y_test = y_test.argmax(-1).flatten().data.cpu().tolist()
            
            pred_test.extend(temp_theta_mean)
            label_test.extend(temp_y_test)
        
        accuracy_test, precision_test, recall_test, f1_score_test = compute_accuracy(pred_test, label_test)
        print ("##################################################")
        print('Epoch {}, loss={}, accuracy_train={}, accuracy_test={}'.format(epoch, loss_epoch / len(input_), accuracy_train, accuracy_test))
        for k in range(num_topic):
            print ("precision_train{}".format(k), "=" , "{:.9f}".format(precision_train[k]), \
                 "recall_train{}".format(k), "=" , "{:.9f}".format(recall_train[k]), \
                 "f1_score_train{}".format(k), "=" , "{:.9f}".format(f1_score_train[k]))
            print ("precision_te{}".format(k), "=" , "{:.9f}".format(precision_test[k]), \
                 "recall_te{}".format(k), "=" , "{:.9f}".format(recall_test[k]), \
                 "f1_score_te{}".format(k), "=" , "{:.9f}".format(f1_score_test[k]))
        emb = model.de_fc.weight.data.detach().cpu().numpy().T
        print_top_words(emb, vocab, 50)
        # print_perp(model)
        write_topwords("results/ursa/{}.txt".format(epoch), emb, vocab, 50)
        print ("##################################################")        

model.eval()
gamma_mean, gamma_logvar = model.gamma()
gm, gl = gamma_mean.data.cpu().numpy(), gamma_logvar.data.cpu().numpy()
print_gamma(gm, seed_words, vocab, vocab2id)

emb = model.de_fc.weight.data.cpu().numpy().T
print_top_words(emb, vocab, 50)
print_perp(model)
