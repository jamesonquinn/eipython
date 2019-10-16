
from collections import defaultdict
import inspect
import pdb

def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

PRINT_COUNTS = defaultdict(lambda: 0)

def isPowerOfTwo (x):

    # First x in the below expression
    # is for the case when x is 0
    return (x>0 and (not(x & (x - 1))) )

def dp(key, *args, do=1, n = 10, break=False):
    global PRINT_COUNTS[key] += 1
    if do:
        if PRINT_COUNTS[key] - n < 1:
            print(key,"line",inspect.currentframe().f_back.f_lineno,PRINT_COUNTS[key],*args)
        elif isPowerOfTwo(PRINT_COUNTS[key]):
            print(key,"line",inspect.currentframe().f_back.f_lineno,PRINT_COUNTS[key])

    if break:
        pdb.set_trace()

def getDebugCount(key):
    return PRINT_COUNTS[key]

def resetDebugCounts():
    for key in PRINT_COUNTS:
        if type(key) == str and key[:4] != "base":
            PRINT_COUNTS[key] = 0
