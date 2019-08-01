from __future__ import print_function

#Like polytopize, but for 2x2.

print('Reloading linearize.')

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


MIN_DIFF = 1e-2

def approx_eq(a,b):
    #print("a",a)
    #print("b",b)
    #print("diff:",torch.abs(torch.add(a, -b)))
    #print("So:",torch.all(torch.lt(torch.abs(torch.add(a, -b)), MIN_DIFF)))
    #print("So2:",torch.all(torch.lt(torch.abs(torch.add(a, -b)), MIN_DIFF))==
    #        torch.all(torch.lt(zs(1),1)))
    return(torch.all(torch.lt(torch.abs(torch.add(a, -b)), MIN_DIFF)))

PERTURBATION = torch.tensor([[1., -1.], [-1.,1.]])

def get_indep(R, C, ns, vs): #note, not-voting is a candidate
    assert len(ns)==R
    assert len(vs)==C
    assert R==2
    assert C==2
    assert torch.all(torch.eq(ns,0.) ^ 1) #`^ 1` means "not".
    assert torch.all(torch.eq(vs,0.) ^ 1) #`^ 1` means "not".
    tot = torch.sum(ns,0)
    #print("sums",tot,sum(vs))
    assert approx_eq(tot,sum(vs)), f'#print("sums",{tot},{sum(vs)})'
    indep = ts([[(rnum * cnum / tot) for cnum in vs] for rnum in ns])#TODO: better pytorch
    mindiag = torch.min(torch.diag(indep))
    zdiag = indep - mindiag * PERTURBATION
    affordance = min(zdiag[0,1],zdiag[1,0]) / 2
    return (zdiag + affordance * PERTURBATION,affordance)

def process_data(data):
    ns, vs = data
    R = len(ns[0])
    C = len(vs[0])
    indeps = [get_indep(R,C,n,v) for n, v in zip(ns, vs)]
    tots = [torch.sum(n) for n in ns]
    print("tots!",tots)
    return (ns, vs, indeps, tots)

def linearize(R, C, raw, preprocessed_data, do_aug=True):
    midpoint, affordance = preprocessed_data
    correction = affordance * (2 / (1 + torch.exp(-raw[0,0])) - 1)
    return midpoint + correction * PERTURBATION

def llinearize(R, C, raw, preprocessed_data, do_aug=True):
    return raw * 2.

def dellinearize(R, C, poly, preprocessed_data):
    return poly / 2.


def delinearize(R, C, poly, preprocessed_data):
    midpoint, affordance = preprocessed_data
    diff = poly[0,0] - midpoint[0,0]
    p = ((diff / affordance) + 1) / 2
    try:
        assert p >= 0
        assert p < 1
    except:
        print("delinearize",poly,preprocessed_data,diff,p)
        raise
    return torch.log(p / (1 - p)).view(1,1)

#
# def dummyPrecinct(R, C, i=0, israndom=True):
#
#     if israndom:
#         ns = pyro.distributions.Exponential(1.).sample(torch.Size([R]))
#         vs = pyro.distributions.Exponential(1.).sample(torch.Size([C]))
#     else:
#         #print("Not random")
#         ns = ts([r+i+1. for r in range(R)])
#         vs = ts([c+i+2. for c in range(C)])
#     #print("ttots:",ns,vs,torch.sum(ns),torch.sum(vs))
#     vs = vs / torch.sum(vs) * torch.sum(ns)
#     #print("ttots:",ns,vs)
#     indep = get_indep(R,C,ns,vs)
#     return(ns,vs,indep)
#
# def test_funs(R, C, innerReps=2, outerReps=2, israndom=True):
#     for i in range(outerReps):
#         ns,vs,indep = dummyPrecinct(R,C,i,israndom)
#         for j in range(innerReps):
#             loc = pyro.distributions.Normal(0.,4.).sample(torch.Size([R-1,C-1]))
#             polytopedLoc = linearize(R,C,loc,indep)
#             try:
#                 assert approx_eq(ns, torch.sum(polytopedLoc, dim=1).view(R)) , "ns fail"
#                 assert approx_eq(vs, torch.sum(polytopedLoc, dim=0).view(C)) , "vs fail"
#                 assert torch.all(torch.ge(polytopedLoc,0)) , "ge fail"
#             except Exception as e:
#                 print(e)
#                 print("loc",loc)
#                 print("polytopedLoc",polytopedLoc)
