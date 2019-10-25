
from collections import defaultdict
import inspect
import pdb
import torch

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
