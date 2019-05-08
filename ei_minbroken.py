#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
print('Yes, I will run.')

from importlib import reload
import go_or_nogo
import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from matplotlib import pyplot as plt
from cmult import CMult
import polytopize
reload(polytopize)
from polytopize import get_indep, polytopize, depolytopize, to_subspace
from pyro import poutine
import myhessian
import numpy as np
import cProfile as profile
ts = torch.tensor


torch.manual_seed(478301986) #Gingles

pyro.enable_validation(True)
pyro.set_rng_seed(0)

# Suppose we have aggregate level counts of two classes of voters and of votes in each of P precincts. Let's generate some fake data.

#...

# Now suppose precinct-level behavior follows a Beta distribution with class-dependent parameters, so that individual level behavior is Bernoulli distributed. We can write this as a generative model that we'll use both for data generation and inference.

MINIMAL_DEBUG = False
if MINIMAL_DEBUG:
    pyrosample = lambda x,y,infer=None : y.sample()
else:
    #
    def pyrosample2(name,d,*args,**kwargs):
        result = pyro.sample(name,d,*args,**kwargs)
        print(f"pyrosample2!!!! {name},{result.size()}")
        return result
    def pyrosample(name,d,*args,**kwargs):
        result = pyro.sample(name,d,*args,**kwargs)
        print(f"#####pyrosample {name},{result.size()}") #
        return result

def model(data=None, scale=1., include_nuisance=False, do_print=False):
    print("model:begin")
    P,R,C = 5,4,3
    all_ps = range(P) #plate(P)
    if include_nuisance:
        eprc_list = []
        with poutine.scale(scale=scale):
            for p in all_ps:
                eprc_list.append(
                    pyrosample2(f'eprc_{p}', dist.Normal(torch.zeros(R,C),3.).to_event(2))
                    )

                if p==(P-1):
                    print(f"model eprc_{p} {eprc_list[0].size()}")
        eprc = torch.stack(eprc_list,0)
    else:
        eprc = ts(0.) #dummy for print statements. TODO:remove

    print("model:end")




# Let's now write a variational approximation.

init_narrow = 10  # Numerically stabilize initialization.
BASE_PSI =.01

def infoToM(Info,psi):
    tlen = len(psi)
    try:
        assert len(Info)==tlen
    except:
        print(Info.size(),tlen)
        raise
    M = []
    for i in range(tlen):
        lseterms = torch.stack([ts(0.),
                            -Info[i,i] + psi[i],
                            -abs(Info[i,i]) + psi[i] +
                                torch.sum(torch.stack([abs(Info[i,j])
                                    for j in range(tlen) if j != i]))])
        M.append( psi[i] * torch.logsumexp(lseterms / psi[i],0))
    return torch.diag(torch.stack(M))

BUNCHFAC = 35
#P=10, BUNCHFAC = 9999: 61/31
#P=10, BUNCHFAC = 1: 110/31
#P=30, BUNCHFAC = 1: 785/31; 2490..1799..1213
#P=30, BUNCHFAC = 1: 675/31; 2490..1799..1213
#P=30, BUNCHFAC = 2: 339/31
#P=30, BUNCHFAC = 3: 96/11
#P=30, BUNCHFAC = 9999: 42/11
#P=30, BUNCHFAC = 9999: 189/51 2346..1708..1175..864..746

ADJUST_SCALE = .05
MAX_NEWTON_STEP = 1.2
RECENTER_PRIOR_STRENGTH = 2.

def recenter_rc(rc):
    rowcentered= (rc - torch.mean(rc,0))
    colcentered = rowcentered - torch.mean(rowcentered,0)
    return colcentered

def guide(data, scale, include_nuisance=False, do_print=False):
    print("guide:begin")

    P,R,C = 5,4,3
    all_ps = range(P) #plate(P)
    for p in all_ps:#pyro.plate('precincts3', P):

        if include_nuisance:
            eprc_adjusted = torch.ones(R,C)
            eprc = pyrosample2(f"eprc_{p}",
                dist.Delta(eprc_adjusted).to_event(2)) #Not adjusting for deviation from hat!!!! TODO: fix!!!!

            if p==(P-1):
                print(f"eprc_{p} adjusted {eprc.size()}")
    print("guide:end")


nsteps = 5001
subset_size = 17
bigprime = 73 #Wow, that's big!


def get_subset(data,size,i):
    if data==None:
        return None,1.
    ns, vs = data
    P = len(ns)
    indices = ts([((i*size + j)* bigprime) % P for j in range(size)])
    subset = (ns.index_select(0,indices) , vs.index_select(0,indices))
    scale = sum(ns) / sum(subset[0]) #likelihood impact of a precinct proportional to n
    #print(f"scale:{scale}")
    return(subset,scale)

def trainGuide():

    # Let's generate voting data. I'm conditioning concentration=1 for numerical stability.

    data = model()
    #print(data)


    # Now let's train the guide.
    svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO())

    pyro.clear_param_store()
    losses = []
    if data==None:
        nsteps = 5
    for i in range(nsteps):
        subset, scale = get_subset(data,subset_size,i)
        print("svi.step(...",i,scale)
        loss = svi.step(subset,scale,True,do_print=(i % 10 == 0))
        losses.append(loss)
        if i % 10 == 0:
            reload(go_or_nogo)
            if data != None:
                go_or_nogo.printstuff(i,loss)
            if go_or_nogo.go:
                pass
            else:
                break

    ##

    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    ##
    pyroStore = pyro.get_param_store()
    for (key, val) in sorted(pyroStore.items()):
        print(f"{key}:\n{val}")

    ns, vs = data
    # print("yhat[0]:",
    #     polytopize(4,3,pyroStore["what_0"],
    #                get_indep(4,3,ns[0],vs[0])))

    return(svi,losses,data)
