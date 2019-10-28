
from collections import defaultdict,Mapping
import inspect
import pdb
import torch
import json
import numpy as np

if False:#use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    TTYPE = torch.float32
else:
    torch.set_default_tensor_type("torch.DoubleTensor")
    TTYPE = torch.float64
    
def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

PRINT_COUNTS = defaultdict(lambda: 0)

def isPowerOfTwo (x):

    # First x in the below expression
    # is for the case when x is 0
    return (x>0 and (not(x & (x - 1))) )

def dp(key, *args, do=1, n = 10, brk=False):
    global PRINT_COUNTS
    PRINT_COUNTS[key] += 1
    if do:
        if PRINT_COUNTS[key] - n < 1:
            print(key,"line",inspect.currentframe().f_back.f_lineno,PRINT_COUNTS[key],":",*args)
        elif isPowerOfTwo(PRINT_COUNTS[key]):
            print(key,"line",inspect.currentframe().f_back.f_lineno,PRINT_COUNTS[key])

    if brk:
        pdb.set_trace()

def getDebugCount(key):
    return PRINT_COUNTS[key]

def resetDebugCounts():
    for key in PRINT_COUNTS:
        if type(key) == str and key[:4] != "base":
            PRINT_COUNTS[key] = 0

def sizes(*l):
    result = ""
    for i, a in enumerate(l):
        nanny =  bool(torch.any(torch.isnan(a)))
        result = result + f"""\n    Size {i}: {list(a.size())}; {"NANny" if nanny else "noNAN"}"""
        if nanny:
            result = result +  f"""\n     """
            for d in range(len(a.size())):
                result = result + "  " + str(d) + ":"
                for j in range(a.size()[d]):
                    if torch.any(torch.isnan(a.index_select(d, torch.tensor([j])))):
                        result = result + "X"
                    else:
                        result = result + "."
    return result

def dets(M):
    result = ""
    for i in range(1,M.size()[0]):
        result = result + f"\n    {i}: {np.linalg.det(M[:i,:i].detach())}"
    return result

def jsonizable(thing):
    #print("jsonizable...",thing)
    #print("type",type(thing), type(thing) is FUCKING_TENSOR_TYPE, type(thing) == type(torch.tensor(1.)))
    try:
        return jsonizable(thing.to_jsonable())
    except Exception as e:
        pass#print(e)
    if isinstance(thing, Mapping):
        dp("jsonizable",[(k,type(t)) for (k,t) in thing.items()])
        return dict([(key, jsonizable(val)) for (key, val) in thing.items()])
    elif type(thing)==list:
        return [jsonizable(a) for a in thing]
    elif torch.is_tensor(thing):
        t = thing.tolist()
        #print("not tense",t)
        return(t)
    return thing

def jsonize(thing):
    #print("jsonizing")
    t = jsonizable(thing)

    #print("jsonizing 2", t)
    return(json.dumps(t, indent=2, sort_keys=True))

    #print("jsonized")
