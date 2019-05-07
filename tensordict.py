#a dictionary-like object whose keys are tensors
import torch

class CachedGradDict(dict):
    def __init__(self, *args, **kw):
        super(TensorDict,self).__init__(*args, **kw)
    def __setitem__(self, k, v):
        super(TensorDict,self).__setitem__(tuple([tuple(item.tolist()) for item in k]),
                                ((torch.tensor(item.tolist()) for item in tg) for item in v))
    def __getitem__(self, k):
        super(TensorDict,self).__getitem__(tuple([tuple(item.tolist()) for item in k]))
