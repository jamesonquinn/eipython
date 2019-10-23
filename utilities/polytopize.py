from __future__ import print_function


print('Reloading polytopize.')

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

from utilities.debugGizmos import *

MIN_DIFF = 5e-2

def approx_eq(a,b):
    #print("a",a)
    #print("b",b)
    #print("diff:",torch.abs(torch.add(a, -b)))
    #print("So:",torch.all(torch.lt(torch.abs(torch.add(a, -b)), MIN_DIFF)))
    #print("So2:",torch.all(torch.lt(torch.abs(torch.add(a, -b)), MIN_DIFF))==
    #        torch.all(torch.lt(zs(1),1)))
    return(torch.all(torch.lt(torch.abs(torch.add(a.float(), -b.float())), MIN_DIFF)))

def get_indep(R, C, ns, vs): #note, not-voting is a candidate
    assert len(ns)==R
    assert len(vs)==C
    assert torch.all(torch.eq(ns,0.) ^ 1) #`^ 1` means "not".
    assert torch.all(torch.eq(vs,0.) ^ 1) #`^ 1` means "not".
    tot = torch.sum(ns,0)
    #print("sums",tot,sum(vs))
    assert approx_eq(tot,sum(vs)), f'#print("sums",{tot},{sum(vs)})'
    indep = ts([[(rnum * cnum / tot) for cnum in vs] for rnum in ns])#TODO: better pytorch
    return indep

def process_data(data):
    ns, vs = data
    R = len(ns[0])
    C = len(vs[0])
    indeps = [get_indep(R,C,n,v) for n, v in zip(ns, vs)]
    tots = [torch.sum(n) for n in ns]
    return (ns, vs, indeps, tots)

def process_dataU(data):
    ns, vs, indeps, tots = process_data(data)
    indepsU = torch.stack(indeps).view(-1,torch.numel(indeps[0]))
    totsU = torch.stack(tots)
    return (ns, vs, indepsU, totsU)


def to_subspace(raw, R, C, ns, vs):
    vdiffs = vs - torch.sum(raw,0)
    tot = torch.sum(vs)
    result = raw + torch.stack([vdiffs*ns[r]/tot for r in range(R)],0)
    try:
        assert approx_eq(torch.sum(result,0), vs)
        #assert approx_eq(torch.sum(result,1), ns)
    except:
        print(f"to_subspace error:{torch.sum(raw,0)}")
        print(f"to_subspace error:{torch.sum(result,0)}")
        print(f"to_subspace error:{vs}")
        print(f"to_subspace error:{(torch.sum(result,1), ns)}")
        print(f"to_subspace error:{vdiffs}")
        print(f"to_subspace error:{tot}")
        print(f"to_subspace error:{vdiffs[0]*vs[0]/tot}")
        print(f"to_subspace error:{torch.stack([vdiffs*ns[r]/tot for r in range(R)],0)}")
        print(f"to_subspace error:{vdiffs}")
        print(f"to_subspace error:{vdiffs}")
        raise
    return(result) #TODO: better pytorch way to do this

def polytopize(R, C, raw, start, do_aug=True):
    if do_aug:
        aug1 = torch.cat((raw,-raw.sum(0).unsqueeze(0)),0)
        aug2 = torch.cat((aug1,-aug1.sum(1).unsqueeze(1)),1)
    else:
        aug2 = raw

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


def polytopizeU(R, C, raw, start):
    aug1 = torch.cat((raw,-raw.sum(1).unsqueeze(1)),1)
    aug2 = torch.cat((aug1,-aug1.sum(2).unsqueeze(2)),2).view(-1,R*C)

    try:
        ratio = torch.div(aug2, -start)
    except:
        print(f"line 67:{R},{C},{raw.size()},{start.size()}")
        raise
    closest = torch.argmax(ratio, 1)

    all_shrinkfacs = start * (1 - torch.exp(-ratio)) / aug2
    #dp("Sizes..",[a.size() for a in (raw,start,aug1,aug2,ratio,all_shrinkfacs,closest.unsqueeze(1))])
    correct_shrinkfacs = all_shrinkfacs.gather(1,closest.unsqueeze(1))
    #print("ptope",start,closest//C,closest%C,
    #        ratio[closest//C,closest%C],aug2[closest//C,closest%C])
    #print(aug2,closest//C,closest%C,r,shrinkfac)
    #print("shrink:",shrinkfac)
    return (start - correct_shrinkfacs * aug2).view(-1,R,C)

DEPOLY_EPSILON = 1e-9
def depolytopizeU(R, C, rawpoly, start):
    poly = rawpoly.view(-1,R*C)
    assert poly.size() == start.size(), f"depoly fail {R},{C},{poly.size()},{start.size()}"
    rawdiff = poly - start
    diff = rawdiff + (rawdiff == 0).float() * DEPOLY_EPSILON
    ratio = torch.div(diff, -start)
    closest = torch.argmax(ratio, 1)
    #r = start.gather(1,closest.unsqueeze(1))
    facs = start * torch.log(1-ratio) / diff

    result = facs.gather(1,closest.unsqueeze(1)) * diff
    #print("depo",fac, ratio[closest//C,closest%C])
    if torch.any(torch.isnan(result)):
        print("depolytopize fail")
        print(R, C, poly[:3,], start[:3,])
        print(diff[:3,],ratio[:3,],closest[:3,])
        print("2depolytopize fail")
        #print(r,facs,result)
        #print(1-ratio[closest//C,closest%C])
        #print(diff[closest//C,closest%C])
    return result.view(-1,R,C)[:,:(R-1),:(C-1)]


def depolytopize(R, C, poly, start):
    assert poly.size() == start.size(), f"depoly fail {R},{C},{poly.size()},{start.size()}"
    diff = poly - start
    ratio = torch.div(diff, -start)
    closest = torch.argmax(ratio)
    r = start[closest//C,closest%C]
    fac = r * torch.log(1-ratio[closest//C,closest%C]) / diff[closest//C,closest%C]

    result = fac * diff
    #print("depo",fac, ratio[closest//C,closest%C])
    if torch.any(torch.isnan(result)):
        print("depolytopize fail")
        print(R, C, poly, start)
        print(diff,ratio,closest)
        print("2depolytopize fail")
        print(r,fac,result)
        print(1-ratio[closest//C,closest%C])
        print(diff[closest//C,closest%C])
    return result[:(R-1),:(C-1)]

def dummyPrecinct(R, C, i=0, israndom=True):

    if israndom:
        ns = pyro.distributions.Exponential(1.).sample(torch.Size([R]))
        vs = pyro.distributions.Exponential(1.).sample(torch.Size([C]))
    else:
        #print("Not random")
        ns = ts([r+i+1. for r in range(R)])
        vs = ts([c+i+2. for c in range(C)])
    #print("ttots:",ns,vs,torch.sum(ns),torch.sum(vs))
    vs = vs / torch.sum(vs) * torch.sum(ns)
    #print("ttots:",ns,vs)
    indep = get_indep(R,C,ns,vs)
    return(ns,vs,indep)

def test_funs(R, C, innerReps=2, outerReps=2, israndom=True):
    for i in range(outerReps):
        ns,vs,indep = dummyPrecinct(R,C,i,israndom)
        for j in range(innerReps):
            loc = pyro.distributions.Normal(0.,4.).sample(torch.Size([R-1,C-1]))
            polytopedLoc = polytopize(R,C,loc,indep)
            try:
                assert approx_eq(ns, torch.sum(polytopedLoc, dim=1).view(R)) , "ns fail"
                assert approx_eq(vs, torch.sum(polytopedLoc, dim=0).view(C)) , "vs fail"
                assert torch.all(torch.ge(polytopedLoc,0)) , "ge fail"
            except Exception as e:
                print(e)
                print("loc",loc)
                print("polytopedLoc",polytopedLoc)
