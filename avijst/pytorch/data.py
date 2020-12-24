import numpy as np
from sklearn import metrics
import math
from typing import *
import torch
from torch.utils.data import Dataset

from typing import *

# fastai utility
def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def compose(x, funcs, *args, **kwargs):
    for f in listify(funcs): 
        x = f(x, **kwargs)
    return x

# class Onehotify():
#     def __init__(self, vocab_size):
#         self.vocab_size = vocab_size
#         self.tokenizer = Tokenizer(num_words=vocab_size)
#     def __call__(self, item):
#         return self.tokenizer.sequences_to_matrix([item], mode='binary')

# class Padify():
#     def __init__(self, maxlen):
#         self.maxlen = maxlen
#     def __call__(self, item):
#         return sequence.pad_sequences([item], maxlen=self.maxlen)

# class YOnehotify():
#     def __init__(self, num_classes):
#         self.num_classes = num_classes
#     def __call__(self, item):
#         categorical = np.zeros((1, self.num_classes))
#         categorical[0, item] = 1
#         return categorical

def collate(b):
    x, y = zip(*b)
    return torch.stack(x), torch.stack(y)

def collate_imdb(b):
    xr, xl, x, y = zip(*b)
    return torch.stack(xr), torch.stack(xl), torch.stack(x), torch.stack(y)

class IdifyAndLimitedVocab():
    def __init__(self, vocab2id, limited_vocab):
        self.vocab2id = vocab2id
        self.limited_vocab = limited_vocab
    def __call__(self, item):
        idlist = [self.vocab2id[w] for w in item if self.vocab2id[w] < self.limited_vocab]
        return np.array(idlist)
    
class Padify():
    def __init__(self, maxlen, pad_id=0):
        self.maxlen, self.pad_id = maxlen, pad_id
    
    def __call__(self, item):
        len_item = len(item)
        if (len_item < self.maxlen):
            item = item + [self.pad_id] * (self.maxlen - len_item)
        return item
    
class Lengthen():
    def __call__(self, item):
        len_item = len(item)
        return len_item
    
class Numpyify():
    def __call__(self, item):
        return np.array(item)

class Onehotify():
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    def __call__(self, item):
        return np.array(np.bincount(item.astype('int'), minlength=self.vocab_size))
    
class YToOnehot():
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def __call__(self, item):
        categorical = [0] * self.num_classes
        categorical[item] = 1
        return categorical

class Tensorify():
    def __call__(self, item):
        return torch.from_numpy(item)

class Floatify():
    def __call__(self, item):
        return item.float()
    
class Longify():
    def __call__(self, item):
        return item.long()
    
class Cpuify():
    def __call__(self, item):
        return item.cpu()
    
class ToInt64():
    def __call__(self, item):
        return item.to(torch.int64)
    
class CheckAndCudify():
    def __init__(self):
        self.ic = torch.cuda.is_available()
    def __call__(self, item):
        return item.cuda() if self.ic else item

class IMDBDataset(Dataset):
    def __init__(self, xr, lenxr, x, y, tfms_xr, tfms_lenxr, tfms_x, tfms_y): 
        self.xr, self.lenxr, self.x, self.y = xr, lenxr, x, y
        self.xr_tfms, self.lenxr_tfms, self.x_tfms, self.y_tfms = tfms_xr, tfms_lenxr, tfms_x, tfms_y
    def __len__(self): 
        return len(self.x)
    def __getitem__(self, i): 
        return compose(self.xr[i], self.xr_tfms), compose(self.lenxr[i], self.lenxr_tfms), \
                compose(self.x[i], self.x_tfms), compose(self.y[i], self.y_tfms)

class Sampler():
    def __init__(self, ds, bs, shuffle=False):
        self.n,self.bs,self.shuffle = len(ds),bs,shuffle
    def __iter__(self):
        self.idxs = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.bs): yield self.idxs[i:i+self.bs]

class DataLoader():
    def __init__(self, ds, sampler, collate_fn=collate):
        self.ds, self.sampler, self.collate_fn = ds, sampler, collate_fn
    def __iter__(self):
        for s in self.sampler: yield self.collate_fn([self.ds[i] for i in s])