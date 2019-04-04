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
    assert approx_eq(tot,sum(cnums))
    indep = ts([[(rnum * cnum / tot) for cnum in cnums] for rnum in rnums])
    return indep



def polytopize(R, C, raw, start):

    aug1 = torch.cat((raw,-raw.sum(0).unsqueeze(0)),0)
    aug2 = torch.cat((aug1,-aug1.sum(1).unsqueeze(1)),1)

    if 0==torch.max(torch.abs(aug2)):
        return(aug2)
    ratio = torch.div(aug2, -start)
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

#print("test 2,2")
test_funs(2,2,israndom=False)
#print("test 5,2")
test_funs(5,2)
#print("test 2,4")
test_funs(2,4)
#print("test 6,8")
test_funs(6,8)
#print("tests done.")

R = 4
C = 3
rnums,cnums,indep = dummyPrecinct(R,C,1,False)

import hessian



#Take an (R-1)*(C-1) point, polytopize it, and compute the pseudo-"Jacobian determinant" (as if the Jacobian were full-ranked)
loc2 = pyro.distributions.Normal(0.,2.).sample(torch.Size([R-1,C-1]))
loc2.requires_grad_(True)
result2 = polytopize(R,C,loc2,indep)
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

import mycatdist
reload(mycatdist)
from mycatdist import MyMultinomial

#print("cm...")
cm3 = dist.Multinomial(3,probs=ts([1.,2.])/3.)
#cm = CMult(3,logits=ts([1.,2.]))
cm2 = CMult(3,probs=ts([1.,2.])/3.)
print("callable?",cm3.__call__)
print("callable?",cm2.__call__)
print("Sampling multinomial:",pyro.sample("toprint",dist.Multinomial(3,probs=ts([1.,2.])/3.)))
print("Sampling cm2:",pyro.sample("toprint",cm2))
#print("m1",dist.Multinomial(3,[.5,.5]))
dd = dist.Multinomial(100, torch.tensor([ 1., 1., 1., 1.]))
#print(pyro.sample("wtf",dd))
#print("m2",dist.Multinomial(3,ts([.5,.5])))
#print("...cm...")

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
    logittotals = eprc+erc+ec
    probs = zs(P,R,C)
    for p in pyro.plate('precincts', P):
        for r in pyro.plate('rgroups', R):
            tmp = torch.exp(logittotals[p,r])
            cprobs = tmp/torch.sum(tmp)
            n = int(data[p,RNUMS,r].item())
            samp= pyro.sample(f"y_{p}_{r}",
                        CMult(n,cprobs))
            y[p,r] = samp

def eiguide(R,C,data):




    yparam = pyro.param("y", y)









mod = eimodel(2,3,ts([[[4,5,1e6],[2,3,4]]]))

sdrc, ec = ts(1.), ts([-.2,0,.2])


sdrc.requires_grad_(True)
ec.requires_grad_(True)
modvals = pyro.condition(mod, data={"sdrc" : sdrc, "ec" : ec})


def coinModel(data):
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    # sample f from the beta prior
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
    # loop over the observed data
    for i in range(len(data)):
        # observe datapoint i using the bernoulli
        # likelihood Bernoulli(f)
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])

# create some data with 6 observed heads and 4 observed tails
coinData = []
for _ in range(6):
    coinData.append(torch.tensor(1.0))
for _ in range(4):
    coinData.append(torch.tensor(0.0))

latent_fairness = ts(.53)

[t.requires_grad_(True) for t in [latent_fairness]]
coinFit = pyro.condition(coinModel, data={"latent_fairness":latent_fairness})
trace1 = poutine.trace(coinFit)
trace2 = trace1.get_trace(coinData)
loss = -trace2.log_prob_sum()
H = hessian.hessian(loss, [latent_fairness], allow_unused=True)
print(loss,H)
