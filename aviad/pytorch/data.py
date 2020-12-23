import numpy as np
from sklearn import metrics
import math

import torch
from torch.utils.data import Dataset, DataLoader

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

class Floatify():
    def __call__(self, item):
        return item.float()
    
class Cudify():
    def __init__(self):
        self.ic = torch.cuda.is_available()
    def __call__(self, item):
        return item.cuda() if self.ic else item

class Tensorify(object):
    def __call__(self, item):
        return torch.from_numpy(item)

class Onehotify(object):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    def __call__(self, item):
        return np.bincount(item, minlength=self.vocab_size)

class YOnehotify():
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def __call__(self, item):
        categorical = np.zeros((1, self.num_classes))
        categorical[0, item] = 1
        return categorical

class URSADataset(Dataset):
    def __init__(self, x, y, tfms_x, tfms_y): 
        self.x, self.y = x, y
        self.x_tfms, self.y_tfms = tfms_x, tfms_y
    def __len__(self): 
        return len(self.x)
    def __getitem__(self, i): 
        if torch.is_tensor(i):
            i = i.tolist()
        batch_x, batch_y = self.x[i], self.y[i]
        return_x, return_y = [], []
        if self.x_tfms:
            batch_x = self.x_tfms(batch_x)
        if self.y_tfms:
            batch_y = self.y_tfms(batch_y)
        return (batch_x, batch_y)