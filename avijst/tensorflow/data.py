import numpy as np
from sklearn import metrics
import math
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

class Onehotify():
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

class IMDBDataset():
    def __init__(self, x, y, tfms_x, tfms_y): 
        self.x, self.y = x, y
        self.x_tfms, self.y_tfms = tfms_x, tfms_y
    def __len__(self): 
        return len(self.x)
    def _get_transform(self, i, tfms):
        return compose(i, tfms)
    def __getitem__(self, i): 
        batch_x, batch_y = self.x[i], self.y[i]
        return_x, return_y = [], []
        if isinstance(i, slice): 
            return_x = [self._get_transform(o, self.x_tfms) for o in batch_x]
        if isinstance(i, slice):
            return_y = [self._get_transform(o, self.y_tfms) for o in batch_y]
        return np.vstack(return_x), np.vstack(return_y)

class DataLoader():
    def __init__(self, ds, bs, drop_last=True): self.ds, self.bs, self.drop_last = ds, bs, drop_last
    def __iter__(self):
        length = len(self.ds) // self.bs if self.drop_last else math.ceil(len(self.ds) / self.bs)
        for i in range(0, length, 1):
            yield self.ds[(i*self.bs):(i*self.bs)+self.bs]