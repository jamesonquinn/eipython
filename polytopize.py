from __future__ import print_function


print('Yes, I will run.')

from importlib import reload



import os
from collections import defaultdict
import numpy as np
import scipy.stats
import torch
ts = torch.tensor
mt = torch.empty
zs = torch.zeros
from torch.distributions import constraints
from matplotlib import pyplot
#matplotlib inline

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('0.3.0')
pyro.enable_validation(True)

MIN_DIFF = 1e-4

def approx_eq(a,b):
    #print("a",a)
    #print("b",b)
    #print("diff:",torch.abs(torch.add(a, -b)))
    #print("So:",torch.all(torch.lt(torch.abs(torch.add(a, -b)), MIN_DIFF)))
    #print("So2:",torch.all(torch.lt(torch.abs(torch.add(a, -b)), MIN_DIFF))==
    #        torch.all(torch.lt(zs(1),1)))
    return(torch.all(torch.lt(torch.abs(torch.add(a, -b)), MIN_DIFF)))

def get_indep(R, C, rnums, cnums): #note, not-voting is a candidate
    assert len(rnums)==R
    assert len(cnums)==C
    assert torch.all(torch.eq(rnums,0.) ^ 1) #`^ 1` means "not".
    assert torch.all(torch.eq(cnums,0.) ^ 1) #`^ 1` means "not".
    tot = sum(rnums)
    #print("sums",tot,sum(cnums))
    assert approx_eq(tot,sum(cnums)), f'#print("sums",{tot},{sum(cnums)})'
    indep = ts([[(rnum * cnum / tot) for cnum in cnums] for rnum in rnums])
    return indep



def polytopize(R, C, raw, start):

    aug1 = torch.cat((raw,-raw.sum(0).unsqueeze(0)),0)
    aug2 = torch.cat((aug1,-aug1.sum(1).unsqueeze(1)),1)

    if 0==torch.max(torch.abs(aug2)):
        return(aug2)
    try:
        ratio = torch.div(aug2, -start)
    except:
        print(f"line 67:{R},{C},{raw.size()},{start.size()}")
        raise
    closest = torch.argmax(ratio)
    r = start[closest//C,closest%C]
    shrinkfac = r * (1 - torch.exp(-ratio[closest//C,closest%C])) / aug2[closest//C,closest%C]
    #print("ptope",start,closest//C,closest%C,
    #        ratio[closest//C,closest%C],aug2[closest//C,closest%C])
    #print(aug2,closest//C,closest%C,r,shrinkfac)
    #print("shrink:",shrinkfac)
    return start - shrinkfac * aug2

def depolytopize(R,C,poly,start):
    diff = poly - start
    ratio = torch.div(diff, -start)
    closest = torch.argmax(ratio)
    r = start[closest//C,closest%C]
    fac = r * torch.log(1-ratio[closest//C,closest%C]) / diff[closest//C,closest%C]

    result = fac * diff
    #print("depo",fac, ratio[closest//C,closest%C])
    return result[:(R-1),:(C-1)]


def dummyPrecinct(R, C, i=0, israndom=True):

    if israndom:
        rnums = pyro.distributions.Exponential(1.).sample(torch.Size([R]))
        cnums = pyro.distributions.Exponential(1.).sample(torch.Size([C]))
    else:
        #print("Not random")
        rnums = ts([r+i+1. for r in range(R)])
        cnums = ts([c+i+2. for c in range(C)])
    #print("ttots:",rnums,cnums,torch.sum(rnums),torch.sum(cnums))
    cnums = cnums / torch.sum(cnums) * torch.sum(rnums)
    #print("ttots:",rnums,cnums)
    indep = get_indep(R,C,rnums,cnums)
    return(rnums,cnums,indep)

def test_funs(R, C, innerReps=2, outerReps=2, israndom=True):
    for i in range(outerReps):
        rnums,cnums,indep = dummyPrecinct(R,C,i,israndom)
        for j in range(innerReps):
            loc = pyro.distributions.Normal(0.,4.).sample(torch.Size([R-1,C-1]))
            polytopedLoc = polytopize(R,C,loc,indep)
            try:
                assert approx_eq(rnums, torch.sum(polytopedLoc, dim=1).view(R)) , "rnums fail"
                assert approx_eq(cnums, torch.sum(polytopedLoc, dim=0).view(C)) , "cnums fail"
                assert torch.all(torch.ge(polytopedLoc,0)) , "ge fail"
            except Exception as e:
                print(e)
                print("loc",loc)
                print("polytopedLoc",polytopedLoc)
