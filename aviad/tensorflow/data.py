import numpy as np
from sklearn import metrics

def compose(x, funcs, *args, **kwargs):
    for f in listify(funcs): 
        x = f(x, **kwargs)
    return x

class URSADataset():
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