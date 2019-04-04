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

def get_directions(R, C, rnums, cnums): #note, not-voting is a candidate
    assert len(rnums)==R
    assert len(cnums)==C
    tot = sum(rnums)
    #print("sums",tot,sum(cnums))
    assert approx_eq(tot,sum(cnums))
    indep = ts([[(rnum * cnum / tot) for cnum in cnums] for rnum in rnums])
    basis = mt(R,C,R,C) #the first R,C tells "which basis vector"; the second R,C hold the elements of that "basis vector"
    for rbas in range(R):
        for cbas in range(C):
            cor = basis[rbas,cbas,rbas,cbas] = indep[rbas,cbas]
            cfac = cor / (cnums[cbas] - cor)
            rfac = cor / (rnums[rbas] - cor)
            for r in range(R): #fill in cbas
                if r==rbas:
                    continue
                basis[rbas,cbas,r,cbas] = -cfac * indep[r,cbas]
            for c in range(C): #fill in rbas
                if c==cbas:
                    continue
                basis[rbas,cbas,rbas,c] = -rfac * indep[rbas,c]
                for r in range(R):
                    if r==rbas:
                        continue
                    basis[rbas,cbas,r,c] = basis[rbas,cbas,r,cbas] * basis[rbas,cbas,rbas,c] / cor
            try:
                assert approx_eq(torch.sum(basis[rbas,cbas], dim=0),0),"rows!"
                assert approx_eq(torch.sum(basis[rbas,cbas], dim=1),0),"cols!"
            except Exception as e:
                print(e)
                print("cols:",torch.sum(basis[rbas,cbas], dim=1))
    return(indep,basis)

def inbasis(R, C, raw, basis, minusone=False):
    result = zs(R, C)
    d = 1 if minusone else 0
    for r in range(R-d):
        for c in range(C-d):
            result.add_(raw[r,c] * basis[r,c]) #scalar times matrix
    return result

def makeBasisTrue(R,C,basis):
    trueBasis = basis.clone()
    for r in range(R):
        trueBasis[r,C-1] = 0.
        trueBasis[r,C-1,r,C-1] = 1.
    for c in range(C-1):
        trueBasis[R-1,c] = 0.
        trueBasis[R-1,c,R-1,c] = 1.
    return(trueBasis)



def polytopize(R, C, raw, basis, start, minusone=False):
    if 0==torch.max(torch.abs(raw)):
        return(start)
    step1 = inbasis(R, C, raw, basis, minusone)
    ratio = torch.div(step1, start)
    closest = torch.min(ratio)
    #print("nums",ratio,closest)
    return(start + ((step1 / -closest) * (1 - torch.exp(closest))))

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
    indep, basis = get_directions(R,C,rnums,cnums)
    return(rnums,cnums,indep,basis)

def test_funs(R, C, innerReps=2, outerReps=2, israndom=True):
    for i in range(outerReps):
        rnums,cnums,indep,basis = dummyPrecinct(R,C,i,israndom)
        for j in range(innerReps):
            loc = pyro.distributions.Normal(0.,4.).sample(torch.Size([R,C]))
            polytopedLoc = polytopize(R,C,loc,basis,indep)
            #print("rnums",rnums)
            #print("polytopedLoc",polytopedLoc)
            #print("basis",basis[0,0])
            try:
                assert approx_eq(rnums, torch.sum(polytopedLoc, dim=1).view(R)) , "rnums fail"
                assert approx_eq(cnums, torch.sum(polytopedLoc, dim=0).view(C)) , "cnums fail"
                assert torch.all(torch.ge(polytopedLoc,0)) , "ge fail"
            except Exception as e:
                print(e)
                print("loc",loc)
                print("polytopedLoc",torch.min(polytopedLoc))
