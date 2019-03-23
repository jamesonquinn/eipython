from __future__ import print_function

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

print("test 2,2")
test_funs(2,2,israndom=False)
print("test 5,2")
test_funs(5,2)
print("test 2,4")
test_funs(2,4)
print("test 6,8")
test_funs(6,8)
print("tests done.")

R = 4
C = 2
rnums,cnums,indep,basis = dummyPrecinct(R,C,1,False)

import hessian

loc = pyro.distributions.Normal(0.,4.).sample(torch.Size([R,C]))
loc.requires_grad_(True)
result = polytopize(R,C,loc,basis,indep)
result_ = torch.sum(result)
#print("Hessian:",hessian.hessian(result_,loc))

#Take an (R-1)*(C-1) point, polytopize it, and compute the pseudo-"Jacobian determinant" (as if the Jacobian were full-ranked)
loc2 = pyro.distributions.Normal(0.,4.).sample(torch.Size([R-1,C-1]))
loc2.requires_grad_(True)
result2 = polytopize(R,C,loc2,basis,indep,True)
j = hessian.jacobian(result2,loc2)
jsquare = zs(R*C,R*C)
RC1 = (R-1)*(C-1)
jsquare[0:R*C,0:RC1] = j
for i in range(R+C-1):
    jsquare[i+RC1,i+RC1] = 1.
#print(jsquare[2:5,2:5])
#print("Jacobian:",j.size(),torch.det(jsquare))

RNUMS = 0 #constant for indexing into data
VNUMS = 1 #ditto

import cmult
reload(cmult)
from cmult import CMult

print("cm...")
cm = CMult(3,logits=ts([1.,2.]))
#print("m1",dist.Multinomial(3,[.5,.5]))
dd = dist.Multinomial(100, torch.tensor([ 1., 1., 1., 1.]))
print(pyro.sample("wtf",dd))
print("m2",dist.Multinomial(3,ts([.5,.5])))
print("...cm...")

@config_enumerate
def eimodel(R,C,data):
    # Hyperparams.
    P = len(data)
    sdc = 5
    sdrc = pyro.sample('sdrc', dist.Exponential(.2))
    sdprc = pyro.sample('sdprc', dist.Exponential(.2))

    with pyro.plate('candidates', C):
        ec = pyro.sample('ec', dist.Normal(0,sdc))
        with pyro.plate('rgroups', R):
            erc = pyro.sample('erc', dist.Normal(0,sdrc))
            with pyro.plate('precincts', P):
                eprc = pyro.sample('eprc', dist.Normal(0,sdprc))

    #with pyro.plate('precinctdata', len(data)) as p:
    #    with pyro.plate('rdata', R) as rs:
            #print("unn", unnormalized)
    #        print("rs",[ts([torch.exp(ec[c] + erc[r,c] + eprc[p,r,c]) for c in range(C)]) for r in rs])
    y = zs(P,R,C)
    probs = zs(P,R,C)
    with pyro.plate('yprecincts', P) as ps:
        with pyro.plate('yrgroups', R) as rs:
            for p in ps:
                for r in rs:
                    for c in range(C):
                        temp = torch.exp(ec[c] + erc[r,c] + eprc[p,r,c])
                        print("eimod innnnnnner",ec.size(),erc.size(),eprc.size(),
                                    ec[c],erc[r,c],eprc[p,r,c])
                        probs[p,r,c] = temp.item()
                    #
                    print("eimod inner",p.item(),r.item(),c,probs[p,r])
                    print("eimod inner2",int(data[p,RNUMS,r].item()))
                    print("eimod inner3",probs[p,r]/torch.sum(probs[p,r]))

                    cprobs = probs[p,r]/torch.sum(probs[p,r])
                    n = int(data[p.item(),RNUMS,r.item()].item())
                    print("eimod inner4",n, cprobs)
                    samp= pyro.sample('y',
                                dist.Multinomial(n,cprobs))
                    print("ei",samp)
                    print("ei2",p,r,y[p,r],"and",y[p,r,0])
                    y[p,r] = samp
    print("eimod",[[int(data[p,RNUMS,r].item()) for r in range(R)] for p in range(len(data))])
    r,p = 0,0
    print("eimod2",[[ts([torch.exp(ec[c] + erc[r,c] + eprc[p,r,c]).item() for c in range(C)])
             for r in range(R)] for p in range(len(data))])
    y = ts([[pyro.sample('y',dist.Multinomial(int(data[p,RNUMS,r].item()),
              ts([torch.exp(ec[c] + erc[r,c] + eprc[p,r,c]).item() for c in range(C)])
            )) for r in range(R)] for p in range(len(data))])

    print("eimodel",y)

eimodel(2,3,ts([[[4,5,1e6],[2,3,4]]]))
