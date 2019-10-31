#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from utilities.debugGizmos import *
dp('base:Yes, I will run.')

from importlib import reload
import csv
import time
import math
import os
import random
import json
import contextlib
from itertools import chain
import cProfile as profile
from collections import OrderedDict, defaultdict


from matplotlib import pyplot as plt
import numpy as np
import pandas

import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro import poutine


from utilities.decorator import reify
from utilities import myhessian
from utilities.rank1torch_vectorized import optimize_Q_objectly, transpose
from utilities import go_or_nogo
from utilities.cmult import CMult
from utilities import polytopize
reload(polytopize)
from utilities.polytopize import get_indep, polytopizeU, depolytopizeU, to_subspace, process_data, approx_eq
from utilities.arrowhead_precision import *

use_cuda = torch.cuda.is_available()



ts = torch.tensor


torch.manual_seed(478301986) #Gingles

pyro.enable_validation(True)
pyro.set_rng_seed(0)


EI_VERSION = "1.2.a0"
init_narrow = 10  # Numerically stabilize initialization.


BUNCHFAC = 35
#P=10, BUNCHFAC = 9999: 61/31
#P=10, BUNCHFAC = 1: 110/31
#P=30, BUNCHFAC = 1: 785/31; 2490..1799..1213
#P=30, BUNCHFAC = 1: 675/31; 2490..1799..1213
#P=30, BUNCHFAC = 2: 339/31
#P=30, BUNCHFAC = 3: 96/11
#P=30, BUNCHFAC = 9999: 42/11
#P=30, BUNCHFAC = 9999: 189/51 2346..1708..1175..864..746

MAX_NEWTON_STEP = 1. #currently, just taking this much of a step, hard-coded
STARPOINT_AS_PORTION_OF_NU_ESTIMATE = 1.
NEW_DETACHED_FRACTION = .1 #as in Newton, get it?
SDS_TO_REDUCE_BY = 1.
SDS_TO_SHRINK_BY = .75



NSTEPS = 5000
SUBSET_SIZE = 20
BIG_PRIME = 73 #Wow, that's big!

FAKE_VOTERS_PER_RACE = 1.
FAKE_VOTERS_PER_REAL_PARTY = .5 #remainder go into nonvoting party

BASE_PSI = .01

QUICKIE_SAVE = (NSTEPS < 20) #save subset; faster
CUTOFF_WINDOW = 500
EXP_RUNNING_MEAN_WINDOW = 150


SIM_SIGMA_NU = .05

PSEUDOVOTERS_PER_CELL = 1.

DEBUG_ARROWHEAD = False

SDRC_VAR = 1.5
SDRC_MEAN = 1.

SDPRC_VAR = 1.5
SDPRC_MEAN = -2.

def toTypeOrNone(t,atype=TTYPE):
    return t.type(atype) if torch.is_tensor(t) else t

class dataWriter:
    def __init__(self,file):
        self.writer = csv.writer(file)

    def writedatarow(self,var,val="",u="",r="",c="",id=""):
        if type(val) is not int:
            try:
                val = float(val)
            except:
                assert val==u"val"
        self.writer.writerow([var,u,r,c,val,id])

    def writeheaderrow(self):
        self.writedatarow(u"var",u"val",u"u",u"r",u"c",u"id")

class EIData:
    def __init__(self,ns,vs,
                    ids = None, ys=None, nus=None,
                    alpha = None, beta =None, sigmanu =None, sigmabeta=None):
        self.ns = toTypeOrNone(ns)
        self.vs = toTypeOrNone(vs)
        self.ys = toTypeOrNone(ys)
        self.nus = toTypeOrNone(nus)
        self.alpha = toTypeOrNone(alpha)
        self.beta = toTypeOrNone(beta)
        self.sigmanu = toTypeOrNone(sigmanu)
        self.sigmabeta = toTypeOrNone(sigmabeta)
        try:
            if ids is None:
                self.ids=list(range(self.U))
            else:
                assert len(ids)==self.U
                self.ids = ids
        except:
            self.ids = None

    @reify
    def ybar(self):
        return torch.sum(self.ys,0)

    @reify
    def tots(self):
        #dp("tots",[(a.dtype,a.device) for a in [self.ns, torch.sum(ns,1)]])
        return torch.sum(self.ns,1)

    @reify
    def indeps(self):
        assert approx_eq(self.tots,torch.sum(self.vs,1)), f'#print("sums",{self.tots},{torch.sum(self.vs,1)})'
        indeps = (torch.matmul(torch.unsqueeze(self.ns + self.C*PSEUDOVOTERS_PER_CELL,-1),
                                torch.unsqueeze(self.vs + self.R*PSEUDOVOTERS_PER_CELL,-2)) /
                    (self.tots + self.R*self.C*PSEUDOVOTERS_PER_CELL).unsqueeze(-1).unsqueeze(-1))
        #dp("indeps",indeps.size())
        return indeps.view(-1,self.R*self.C)

    @reify
    def nns(self): #normalized ns
        #dp("nns",[a.dtype for a in [self.ns, self.tots]])
        return self.ns/self.tots.unsqueeze(1)


    @reify
    def nvs(self): #normalized ns
        return self.vs/self.tots.unsqueeze(1)

    @reify
    def U(self):
        return self.ns.size()[0]

    @reify
    def R(self):
        return self.ns.size()[1]

    @reify
    def C(self):
        try:
            return self.vs.size()[1]
        except:
            return 3

    @reify
    def M(self):
        # M is the matrix of linear constraints on Q (not counting the inequalities)
        # M is (R+C-1)-by-RC
        R,C = self.R,self.C
        M_top = torch.eye(C)
        if R>2:
          M_bottom = torch.cat((torch.ones(1,C),torch.zeros(R-2,C)),0)
        else: #i.e. if R==2
          M_bottom = torch.ones(1,C)
        for r in range(1,R):
            M_top = torch.cat((M_top,torch.eye(C)),1)
            bottom_new = torch.zeros(R-1,C)
            if r<R-1:
                bottom_new[r]=torch.ones(1,C)
            M_bottom = torch.cat((M_bottom, bottom_new),1)
        M = torch.cat((M_top,M_bottom),0)
        return M

    @reify
    def D(self):
        # Matrix D = M^T*(M*M^T)^{-1}*M
            # D is the matrx of projection to orthogonal complement of ker M;
            # it helps us obtain the nearest Q that actually satisfies the linear constraints
        M = self.M
        D = torch.inverse(torch.matmul(M,transpose(M)))
        D=torch.matmul(transpose(M),torch.matmul(D,M))
        return D

    @reify
    def v_d(self):
        return torch.unsqueeze(torch.cat((self.nvs,self.nns),-1),-1)

    @reify
    def nind(self):
        return  torch.matmul(torch.unsqueeze(self.nns,-1),torch.unsqueeze(self.nvs,-2))

    @reify #NOTE: code duplicated in EISubData
    def init_beta_errorQ(self):
        return (torch.ones(self.U,self.C),torch.ones(self.U,self.R,self.C))

    @reify #NOTE: code duplicated in EISubData
    def getStuff(self):
        return (self.U, self.R, self.C, self.ns, self.vs)

    def save(self,filename):
        """
        Note: for both save and load, file format is as follows: (without comments)

        var,u,i,val,id    #field names
        R,,,3,            #value of R — num races
        C,,,3,            #value of C — num "candidates"
        U,,,1000,         #value of U — num precincts
        v,0,0,10,HOKE:p2  #10 votes for cand 0 in precinct 0. also, id for precinct 0; only present in cand 0 line
        v,0,1,10,         #10 votes for cand 1 in precinct 0.
        ...
        v,2,999,33        #33 votes for cand 2 in precinct 999
        n,0,0,50          #50 voters of race 0 in precinct 0
        ....
        n,2,999,111       #111 voters of race 2 in precinct 999

        """
        assert not (os.path.exists(filename)) #don't just blindly overwrite!
        with open(filename,"w", newline="\n") as file:
            writer = dataWriter(file)
            writer.writeheaderrow()
            writer.writedatarow(u"R",self.R)
            writer.writedatarow(u"C",self.C)
            writer.writedatarow(u"U",self.U)
            writer.writedatarow(u"sigmanu",self.sigmanu)
            writer.writedatarow(u"sigmabeta",self.sigmabeta)
            if self.alpha is not None:
                for (c,val) in enumerate(self.alpha):
                    writer.writedatarow(u"alpha",float(val),c=c)
            if self.beta is not None:
                for (r,vals) in enumerate(self.beta):
                    for (c,val) in enumerate(vals):
                        writer.writedatarow(u"beta",float(val),r=r,c=c)
            #
            if self.nus is not None:
                for (u,pvals) in enumerate(self.nus):
                    for (r,vals) in enumerate(pvals):
                        for (c,val) in enumerate(vals):
                            writer.writedatarow(u"nu",float(val),u=u,r=r,c=c)
            #
            if self.ys is not None:
                for (u,pvals) in enumerate(self.ys):
                    for (r,vals) in enumerate(pvals):
                        for (c,val) in enumerate(vals):
                            writer.writedatarow(u"y",float(val),u=u,r=r,c=c)
            #
            for u, (pvs, id) in enumerate(zip(self.vs, self.ids)):
                for c, v in enumerate(pvs):
                    if c==0:
                        writer.writedatarow(u"v",u=u,c=c,val=float(v),id=id)
                    else:
                        writer.writedatarow(u"v",u=u,c=c,val=float(v))
            #
            for u, pns in enumerate(self.ns):
                for r, n in enumerate(pns):
                    writer.writedatarow(u"n",u=u,r=r,val=float(n))

    @classmethod
    def load(cls, filename): #Throws error on failure; use inside a try block.
        with open(filename,"r") as file:
            reader = csv.reader(file)
            header = next(reader)
            vs, ns = (None, None)
            ys = nus = alpha = beta = sigmanu = sigmabeta = ids = None
            for line in reader:
                var, u, r, c, val, id = line
                u, r, c = [int(a) for a in [u or 0,r or 0,c or 0]]
                if var==u"R":
                    ddp("R",type(val),val)
                    R = int(val)
                    continue
                if var==u"C":
                    ddp("C",type(val),val)
                    C = int(val)
                    continue
                if var==u"U":
                    ddp("U",type(val),val)
                    U = int(val)
                    continue
                if var==u"sigmanu":
                    sigmanu = float(val)
                    continue
                if var==u"sigmabeta":
                    sigmabeta = float(val)
                    continue
                if var==u"alpha":
                    if alpha is None:
                        alpha = -torch.ones(C)
                    alpha[c] = float(val)
                    continue
                if var==u"beta":
                    if beta is None:
                        beta = -torch.ones(R,C)
                    beta[r,c] = float(val)
                    continue
                if var==u"nu":
                    if nus is None:
                        nus = -torch.ones(U,R,C)
                    nus[u,r,c] = float(val)
                    continue
                if var==u"y":
                    if ys is None:
                        ys = -torch.ones(U,R,C)
                    ys[u,r,c] = float(val)
                    continue
                if var==u"v":
                    if vs is None:
                        vs = -torch.ones(U,C)
                        ids = [None] * U
                    vs[u,c] = float(val)
                    if id != "":
                        ids[u] = id
                    continue
                if var==u"n":
                    if ns is None:
                        ns = -torch.ones(U,R)
                    ns[u,r] = float(val)
                    continue
            #
            #dp("ns:",ns)
            #dp("vs:",vs)
            try:
                assert torch.all(ns>0) #race counts strictly positive
                assert torch.all(vs+.1>0) #vote counts non-negative
                assert all(ids) #ids all exist
            except Exception as e:
                print("bad file load",e)
        ddp("alpha",alpha)
        ddp("beta",beta)
        data = cls(ns, vs, ids, ys, nus,
                    alpha, beta, sigmanu, sigmabeta)
        return data

#A "decorator"-ish thingy to make it easy to tell EISubData to look it up in full data.
def sub(**kwargs): #I really hate typing quotation marks
    for k,v in kwargs.items():
        if v: #subset
            return reify(lambda self: getattr(self.full,k).index_select(0,self.indices),k)
        return reify(lambda self: getattr(self.full,k),k)
    raise Exception("This shouldn't happen")

class EISubData:
    def __init__(self, full, indices):
        self.full = full
        self.indices = indices

    #BE CAREFUL that variable names match with argument names here; more metaprogramming to make this automatic would be too much trouble
    #Subsetted attributes (=1)
    ns = sub(ns=1)
    vs = sub(vs=1)
    indeps = sub(indeps=1)
    tots = sub(tots=1)
    nns = sub(nns=1)
    nvs = sub(nvs=1)
    v_d = sub(v_d=1)
    nind = sub(nind=1)

    #verbatim attributes (=0)
    M = sub(M=0)
    D = sub(D=0)
    R = sub(R=0)
    C = sub(C=0)

    @reify
    def U(self):
        return len(self.indices)

    @reify
    def init_beta_errorQ(self):
        return (torch.ones(self.U,self.C),torch.ones(self.U,self.R,self.C))

    @reify
    def getStuff(self):
        return (self.U, self.R, self.C, self.ns, self.vs)

    @reify
    def ids(self):
        return [self.full.ids[i] for i in self.indices]

def legible_values(R,C):
    ecraw = torch.zeros(C-1)
    ecraw[0] = .5
    ecraw[1] = -.25
    ec = expand_and_center(ecraw)
    ercraw = torch.zeros(R-1,C-1)
    ercraw[0,0] = -1.
    ercraw[1,0] = -.5
    ercraw[1,1] = 1.
    erc = expand_and_center(ercraw)
    sdc = ec.std()
    sdrc = erc.std()
    ddp("Standard deviations:", sdc,sdrc)
    return (ec, erc)

def model(data=None, scale=1., include_nuisance=True, do_print=False, *args, **kwargs):
    """

    Notes:
        -if data is None, creates 30 precincts of data from nothing
        -otherwise, data should be a tuple (ns, vs, indeps, tots).
            -if vs is None, create & return data from model, with that ns
            -otherwise, model conditions on that vs (ie, use for density, not sampling)
    """
    P, R, C, ns, vs = data.getStuff

    prepare_ps = range(P)
    ps_plate = pyro.plate('all_sampled_ps',P)
    @contextlib.contextmanager
    def all_sampled_ps():
        with ps_plate as p, poutine.scale(scale=scale) as pscale:
            yield p

    sdc = 5
    sdrc = pyro.sample('sdrc', dist.LogNormal(SDRC_MEAN,SDRC_VAR))
    if include_nuisance:
        sdprc = pyro.sample('sdprc', dist.LogNormal(SDPRC_MEAN,SDPRC_VAR))

    if vs is None:
        sdprc = SIM_SIGMA_NU
    #dp(f"sdprc in model:{sdprc}")

    #This is NOT used for data, but instead as a way to sneak a "prior" into the guide to improve identification.
    #param_residual=pyro.sample('param_residual', dist.Normal(0.,1.))


    ec = pyro.sample('ec', dist.Normal(torch.zeros(C),sdc).to_event(1))
    erc = pyro.sample('erc', dist.Normal(torch.zeros(R,C),sdrc).to_event(2))
    if vs is None:
        if data.alpha is None:
            ec,erc = legible_values(R,C)
        else:
            ec,erc = data.alpha, data.beta
            ddp("alpha beta",data.alpha, data.beta)

    if include_nuisance:
        with all_sampled_ps() as p_tensor:
            logits = (
                pyro.sample(f'logits', dist.Normal((ec + erc).expand(P,R,C), sdprc).to_event(2))
                ) #eprc.size() == [P,R,C] because plate dimension happens on left
        #dp("Model: sampling eprc",eprc[0,0,0])
    else:
        logits = torch.zeros(P,R,C) #dummy for print statements. TODO:remove


    # with pyro.plate('candidatesm', C):
    #     ec = pyro.sample('ec', dist.Normal(0,sdc))
    #     with pyro.plate('rgroupsm', R):
    #         erc = pyro.sample('erc', dist.Normal(0,sdrc))
    #         if include_nuisance:
    #             with pyro.plate('precinctsm', P):
    #                 eprc = pyro.sample('eprc', dist.Normal(0,sdprc))

    with all_sampled_ps() as p_tensor:#pyro.plate('precinctsm2', P):
        if vs is None:
            y = torch.zeros(P,R,C)
            for p in p_tensor:
                for r in range(R):
                    #dp("modsize",[a for a in [ns[p,r], logits[p,r]]])
                    try:
                        tmp = dist.Multinomial(int(ns[p,r]),logits=logits[p,r]).sample()
                        #dp(f"y_{p}_{r}: {tmp} {y[p,r]}")
                        y[p,r] = tmp
                    except:
                        ddp("modsize2",p,r,[a for a in [ns[p,r], logits[p,r]]])
                        raise
        else:
            if not torch.all(torch.isfinite(logits)):
                #dp("logits!!!")
                for p in range(P):
                    if not torch.all(torch.isfinite(logits[p])):
                        ddp("nan in logits[p]",p)
                        ddp(logits[p])
                        ddp(ec)
                        ddp(erc)
            y = pyro.sample(f"y",
                        CMult(1000,logits=logits).to_event(1))
                        #dim P, R, C from plate, to_event, CMult
                        #note that n is totally fake — sums are what matter.
                        #TODO: fix CMult so this fakery isn't necessary.
            dp("sampled y",y[0])
            try:
                yy = y - PSEUDOVOTERS_PER_CELL
                assert approx_eq(torch.sum(yy,1),vs)
                assert approx_eq(torch.sum(yy,2),ns)
            except Exception as e:
                print("Unequal...")
                print(torch.sum(yy,1)[1],vs[1])
                #import pdb; pdb.set_trace()
            #dp("model y",scale,y.size(),y[0,:2,:2])
            #dp("ldim",logits.size(),y[0,0,0])

    if vs is None:
        #
        ddp(f"ec:{ec}")
        ddp(f"erc:{erc}")
        ddp(f"y[0]:{y[0]}")
        vs = torch.sum(y,1)
        print("VS",vs[:4])

        return EIData(ns,vs,data.ids,y, logits - ec - erc, ec, erc, sdprc, sdrc)

    ddp("model:end", sizes(ns,vs,ec,erc,logits,y))



def expand_and_center(tens, return_ldaj=False):
    result = tens
    for i,n in enumerate(tens.size()):
        result = torch.cat([result,
                    -torch.sum(result,i).unsqueeze(i)]
                ,i)

    if return_ldaj:
        return (result,0.) #Not actually zero, but constant, so whatevs.
    return result


def get_param(inits,name,default,*args,**kwargs):
    return pyro.param(name,inits.get(name,default),*args,**kwargs)


# This function is to guess a good starting point for eprcstar. But removing for now because:
# 1. SIMPLIFY!
# 2. We're doing 1 (partial) step of Newton's Method anyway so this is mostly redundant.


# def initial_eprc_star_guess(totsp,pi_p,Q2,Q_precision,pi_precision):
#     pi_precision = totsp / pi_p / (torch.ones_like(pi_p) - pi_p)
#     pweighted_Q = pi_p + (Q2-pi_p) * Q_precision / (Q_precision + pi_precision)
#     return (torch.log(pweighted_Q/pi_p))

def exp_ldaj(t,return_ldaj=False):
    result = torch.exp(t)
    if return_ldaj:
        return (result,torch.sum(t))
    return result

def softmax(t,minval=0.,mult=80.):
    if minval == 0.:
        mins = torch.zeros_like(t)
    else:
        mins = torch.ones_like(t)*minval*mult
    return torch.logsumexp(torch.stack((t*mult,mins)),0)/mult

def guide(data, scale, include_nuisance=True, do_print=False, inits=dict(),
            *args, **kwargs):
    ddp("guide:begin",scale,include_nuisance)


    ##################################################################
    # Set up plates / weights
    ##################################################################
    ns, vs, indeps, tots = data.ns, data.vs, data.indeps, data.tots
    P = len(ns)
    R = len(ns[1])
    C = len(vs[1])

    prepare_ps = range(P) #for dealing with stared quantities (no pyro.sample)
    ps_plate = pyro.plate('all_sampled_ps',P)
    @contextlib.contextmanager
    def all_sampled_ps(): #for dealing with unstared quantities (include pyro.sample)
        with ps_plate as p, poutine.scale(scale=scale) as pscale:
            yield p

    ##################################################################
    # Get guide parameters (stars)
    ##################################################################

    gamma_star_data = OrderedDict()
    fstar_data = OrderedDict() #frequentist
    pstar_data = OrderedDict()
    transformation = defaultdict(lambda: lambda x: x) #factory of identity functions

    logsdrcstar = get_param(inits,'logsdrcstar',ts(0.))
    fstar_data.update(sdrc=logsdrcstar)
    transformation.update(sdrc=exp_ldaj)
    if include_nuisance:
        #logsdprcstar = get_param(inits,'logsdprcstar',ts(-3.))
        #fstar_data.update(sdprc=logsdprcstar)
        #transformation.update(sdprc=exp_ldaj)
        eprcstar_startingpoint = torch.zeros(P,R,C,requires_grad =True) #not a pyro param...
        #eprcstar_startingpoint[p].requires_grad_(True) #...so we have to do this manually

    ecstar_raw = get_param(inits,'ecstar_raw', torch.zeros(C-1))
    ercstar_raw = get_param(inits,'ercstar_raw', torch.zeros(R-1,C-1))
    gamma_star_data.update(ec=ecstar_raw,erc=ercstar_raw)
    transformation.update(ec=expand_and_center, erc=expand_and_center)

    # with pyro.plate('candidatesg', C):
    #     ecstar = pyro.param('ecstar', ts(0.))
    #     with pyro.plate('rgroupsg', R):
    #         ercstar = pyro.param('ercstar', ts(0.))
    #         if include_nuisance:
    #             with pyro.plate('precinctsg', P):
    #                 eprcstar = pyro.param('eprcstar', ts(0.))
    #             gamma_star_data.update(eprc=eprcstar)


    ##################################################################
    # Amortize to find ystar and wstar
    ##################################################################

    ec2r = ecstar_raw.detach().requires_grad_()
    erc2r = ercstar_raw.detach().requires_grad_()

    ec2 = expand_and_center(ec2r)
    erc2 = expand_and_center(erc2r)
    #dp("sizes",sizes(ec2r,erc2r,ec2,erc2))


    #Including expand_and_center makes erc not identifiable; this is an issue for writeup, not coding


    #Amortize stars

    logittotals = ec2 + erc2
    #dp("sizes1 ",P,R,C,eprc.size(),ec.size(),erc.size(),logittotals.size())
    if include_nuisance:
        #dp("adding starting point")
        logittotals = logittotals + eprcstar_startingpoint # += doesn't work here because mumble in-place mumble shape
    else:
        logittotals = logittotals.expand(P,R,C)
    pi_raw = torch.exp(logittotals)
    pi = pi_raw / torch.sum(pi_raw,-1).unsqueeze(-1)


    log_jacobian_adjustment = torch.tensor(0.)

    #dp("guide:pre-p")
    if True: #don't deindent yet
    #for p in prepare_ps:#pyro.plate('precinctsg2', P):
        #precalculation - logits to pi


        #dp("amosize",sizes(pi,data.nvs, data.nns))

        Qraw, iters = optimize_Q_objectly(pi,data,tolerance=.01,maxiters=3)


        #get ŷ^(0
        #dp(f"optimize_Q_objectly {p}:{iters}")
        toots = tots.view(P,1,1)
        Q = Qraw + PSEUDOVOTERS_PER_CELL / toots
        ystars = Qraw * toots + PSEUDOVOTERS_PER_CELL #TODO: remove inefficiency but I'm just being extra-careful for now.

        if torch.any(ystars < 0):
            print("ystars < 0")
            import pdb; pdb.set_trace()
        #depolytopize
        wstars = depolytopizeU(R,C,ystars,indeps,lineno())
        wstars_list = [wstar for wstar in wstars] #separate tensor so sparse hessian works... I know, this seems crazy.
        wstars2 = torch.stack(wstars_list)

        ystars2,ldaj = polytopizeU(R,C,wstars2,indeps,return_ldaj=True)
        log_jacobian_adjustment += ldaj


        #get ν̂^(0)
        if include_nuisance:
            QbyR = Q/torch.sum(Q,-1).unsqueeze(-1)
            logresidual_raw = torch.log(QbyR / pi)
            ystars_variance_approx = ystars # 1/(1/ystars + 1/(torch.sum(ystars,2,keepdim=True) - ystars))
            lr_sd_of_like = torch.log((ystars+torch.sqrt(ystars_variance_approx))/ystars)
                #1 sd down, rescaled, logged, minus orig; rough estimate of sd of likelihood of logresidual

            sign_residual = logresidual_raw.sign()
            abs_residual = logresidual_raw * sign_residual
            shrunk_residual = softmax(abs_residual - lr_sd_of_like * SDS_TO_REDUCE_BY) * sign_residual
            sdprc = shrunk_residual.std()

            fstar_data.update(sdprc=sdprc)
            lr_prec_of_like = lr_sd_of_like ** -2 #sd to precision

            eprcstars = STARPOINT_AS_PORTION_OF_NU_ESTIMATE* logresidual_raw*lr_prec_of_like/SDS_TO_SHRINK_BY/(lr_prec_of_like/SDS_TO_SHRINK_BY + 1/sdprc**2)
            if do_print:
                print("sds:",logresidual_raw.std(),sdprc,eprcstars.std())
            #was: initial_eprc_star_guess(tots[p],pi[p],Q2,Q_precision,pi_precision))
            eprcstars_list = [eprcstar for eprcstar in eprcstars]
            eprcstars2 = torch.stack(eprcstars_list)
            #import pdb; pdb.set_trace()


    #dp("w and nu",sizes(ystars,wstars,wstars2,ystars2,eprcstars,eprcstars2))
    if False:#include_nuisance:
        ddp("w and nu 2",torch.sum(ystars-ystars2),
                    torch.sum(eprcstars- eprcstars2),
                    torch.sum(wstars-wstars2))
        ddp("w3", ystars[0], ystars2[0], wstars2[0])

    pstar_data.update(y=ystars2)
    #dp("y is",pstar_data["y"].size(),ystar[-1].size(),ystar[0][0,0],ystars2[0][0,0])
    if include_nuisance:
        logits = expand_and_center(ecstar_raw) + expand_and_center(ercstar_raw) + eprcstars2
    else:

        logits = (expand_and_center(ecstar_raw) + expand_and_center(ercstar_raw)).repeat(P,1,1)
    pstar_data.update(logits=logits)


    #dp("guide:post-p")


    ################################################################################
    # Get hessian(s)
    ################################################################################

    #Start with gamma

    transformed_star_data = OrderedDict()
    for k,v in chain(gamma_star_data.items(),pstar_data.items(),fstar_data.items()):
        if k in transformation:
            transformed_star_data[k],ldaj = transformation[k](v,return_ldaj=True)
            log_jacobian_adjustment += ldaj
        else:
            transformed_star_data[k] = v

    #
    #dp("line ",lineno())
    hess_center = pyro.condition(model,transformed_star_data)
    #dp("line ",lineno())
    mytrace = poutine.block(poutine.trace(hess_center).get_trace)(data, scale, include_nuisance)
    #dp("line ",lineno(),P,R,C)
    log_posterior = mytrace.log_prob_sum()+log_jacobian_adjustment
    ddp("lp: ",log_posterior,log_jacobian_adjustment,lineno())



    hessian_stars_in_sampling_order = [] #this will have eprcstars — good for taking hessian but wrong for mean
    mean_stars_in_sampling_order = [] #this will have logits — right mean, but not upstream, so can't use for hessian
    for part_name in list(gamma_star_data.keys()):
        #dp(f"adding {part_name} to stars_in_sampling_order")
        hessian_stars_in_sampling_order.append(gamma_star_data[part_name])
        mean_stars_in_sampling_order.append(gamma_star_data[part_name])
    gamma_dims = sum(gamma_part.numel() for gamma_part in gamma_star_data.values())
    #dp("len gamma",len(stars_in_sampling_order),stars_in_sampling_order)

    #add pstar_data to stars_in_sampling_order — but don't get it from pstar_data because it comes in wrong format
    if include_nuisance:
        tensors_per_unit = 2 #tensors, not elements=(R-1)*(C-1) + R*C
        #dpu"lam sizes",R,C,sizes(wstars_list[0],eprcstars_list[0]))
        dims_per_unit = (R-1)*(C-1) + R*C
        for w,eprc,logit in zip(wstars_list,
                        eprcstars_list, #This is NOT the center of the distribution; tstar's `logits`
                                #However, the Hessian wrt `eprcstars` should equal that wrt `logits`
                                #And using `logits` here wouldn't work because it's not upstream on the autodiff graph
                        logits):
            hessian_stars_in_sampling_order.extend([w,eprc])
            mean_stars_in_sampling_order.extend([w,logit])
    else:
        tensors_per_unit = 1 #tensors, not elements=(R-1)*(C-1)
        dims_per_unit = (R-1)*(C-1)
        hessian_stars_in_sampling_order.extend(wstars_list) #`wstars_list` is already a list, not a tensor, so this works
        mean_stars_in_sampling_order.extend(wstars_list) #`wstars_list` is already a list, not a tensor, so this works
    full_dims = sum(star.numel() for star in mean_stars_in_sampling_order) #doesn't matter which one, hessian_... would be fine

    #dp("lp:::",log_posterior)
    big_arrow, big_grad = myhessian.arrowhead_hessian_precision(log_posterior,
                    hessian_stars_in_sampling_order, #This is NOT the center of the distribution; that's `logits`
                            #However, the Hessian wrt `eprcstars` should equal that wrt `logits`
                            #And using `logits` here wouldn't work because it's not upstream on the autodiff graph
                    len(gamma_star_data), #tensors, not elements=gamma_dims
                    tensors_per_unit,
                    return_grad=True)


    ##################################################################
    # Put Jacobian into guide density
    ##################################################################
    stupid = torch.exp(-log_jacobian_adjustment)
    if (stupid>0):
        junk = pyro.sample("jacobian",
                        dist.Uniform(torch.zeros(1), stupid),
                        infer={'is_auxiliary': True}) #there's probably a better way but whatevs.
    ##################################################################
    # Sample gamma (globals)
    ##################################################################

    #declare global-level psi params
    globalpsi = get_param(inits,'globalpsi',torch.ones(gamma_dims)*BASE_PSI,
                constraint=constraints.positive)
    #declare precinct-level psi params
    precinctpsi = get_param(inits,'precinctpsi',BASE_PSI * torch.ones(dims_per_unit),
                constraint=constraints.positive)

    #dp"setpsis",sizes(globalpsi,precinctpsi))
    big_arrow.setpsis(globalpsi,precinctpsi)
    big_arrow.weights = [scale] * P

    #head_precision, head_adjustment,  = rescaledSDD(-neg_big_hessian,combinedpsi) #TODO: in-place


    #gamma_info = big_hessian[:gamma_dims,:gamma_dims]

    all_means = torch.cat([tpart.contiguous().view(-1) for tpart in mean_stars_in_sampling_order],0)
    #dp("all_means",all_means)
    #dp(torch.any(torch.isnan(torch.diag(neg_big_hessian))),torch.any(torch.isnan(torch.diag(big_hessian))))
    gamma_mean = all_means[:gamma_dims]
    #dp("detirminants",np.linalg.det(gamma_info.detach()),np.linalg.det(big_hessian.detach()))
    #dp(gamma_info[:3,:3])
    #dp(-neg_big_hessian[:6,:3])

    gamma = pyro.sample('gamma',
                    dist.OMTMultivariateNormal(gamma_mean, torch.cholesky(big_arrow.marginal_gg_cov())),
                    infer={'is_auxiliary': True})
    g_delta = gamma - gamma_mean

    #decompose gamma into specific values
    tmpgamma = gamma
    for pname, pstar in gamma_star_data.items():
        elems = pstar.nelement()
        pdat, tmpgamma = tmpgamma[:elems], tmpgamma[elems:]
        #dp(f"adding {pname} from gamma ({elems}, {pstar.size()}, {tmpgamma.size()}, {pdat})" )

        if pname in transformation:
            pyro.sample(pname, dist.Delta(transformation[pname](pdat.view(pstar.size())))
                                .to_event(len(list(pstar.size())))) #TODO: reshape after transformation, not just before???
        else:
            pyro.sample(pname, dist.Delta(pdat.view(pstar.size()))
                                .to_event(len(list(pstar.size()))))
    assert list(tmpgamma.size())[0] == 0


    for k,v in fstar_data.items():
        pyro.sample(k, dist.Delta(transformation[k](v)))



    ##################################################################
    # Sample lambda_i (locals)
    ##################################################################

    #TODO: uncomment the following, which is fancy logic for learning what fraction of a Newton's method step to take

    #precinct_newton_step_multiplier_logit = pyro.param(
    #        'precinct_newton_step_multiplier_logit',ts(0.))
    #epnsml = torch.exp(precinct_newton_step_multiplier_logit)
    step_mult = MAX_NEWTON_STEP #* epnsml / (1 + epnsml)
    wsamps = []
    logit_samps = []
    adjusted_means = []
    global_indices = ts(range(gamma_dims))
    for p in range(P):


        precinct_indices = ts(range(gamma_dims + p*dims_per_unit, gamma_dims + (p+1)*dims_per_unit))

        #dp(f"gamma_1p_hess:{P},{p},{B},{P//B},{len(big_HWs)},{big_HWs[p//B].size()},")
        #dp(f"HW2:{big_HWs[p//B].size()},{list(full_indices)}")
        conditional_mean, conditional_cov = big_arrow.conditional_ll_mcov(p,g_delta,
                                all_means.index_select(0,precinct_indices))

        precinct_cov = big_arrow.llinvs[p] #for Newton's method, not for sampling #TODO: This is actually the same as conditional_cov; remove?

        precinct_grad = big_grad.index_select(0,precinct_indices) #[gamma_dims + pp*(R-1)*(C-1): gamma_dims + (pp+1)*(R-1)*(C-1)]

        #dp("precinct:::",gamma_1p_hess.size(),precinct_cov.size(),big_grad.size(),precinct_grad.size(),)
        if include_nuisance:
            adjusted_mean_raw = conditional_mean + step_mult * torch.mv(precinct_cov, precinct_grad)
            adjusted_mean = adjusted_mean_raw.detach() * NEW_DETACHED_FRACTION + adjusted_mean_raw * (1 - NEW_DETACHED_FRACTION)
                                 #one (partial, as defined by step_mult) step of Newton's method
                                 #Note: this applies to both ws and nus (eprcs). I was worried about whether that was circular logic but I talked with Mira and we both think it's actually principled.
        else:
            adjusted_mean = conditional_mean


        adjusted_means.append(adjusted_mean)

        try:
            pstuff = pyro.sample(f"pstuff_{p}",
                            dist.OMTMultivariateNormal(adjusted_mean, torch.cholesky(conditional_cov)),
                            infer={'is_auxiliary': True})

        except:
            ddp("error sampling pstuff",p,conditional_cov.size())
            print(conditional_cov)
            rawp = big_arrow.raw_lls[p]
            print(rawp)
            print(f"""dets:{np.linalg.det(conditional_cov.data.numpy())},
                    {np.linalg.det(rawp.data.numpy())},
                    {np.linalg.det(big_arrow.gg.data.numpy())},
                    {np.linalg.det(big_arrow._mgg.data.numpy())},""")
            print("mean",adjusted_mean.size())
            raise
        w_raw = pstuff[:(R-1)*(C-1)]
        wsamps.append(w_raw.view(R-1,C-1))
        #y = polytopize(R,C,w_raw.view(R-1,C-1),indeps[p])
        #ysamps.append(y.view(1,R,C))

        if include_nuisance:
            logit = pstuff[(R-1)*(C-1):].view(1,R,C)
            logit_samps.append(logit)

    with all_sampled_ps():
        ws = torch.stack(wsamps)
        ys = polytopizeU(R,C,ws,indeps)
        if not torch.all(torch.isfinite(ys)):
            for p in range(P):
                if not torch.all(torch.isfinite(ys[p,:,:])):
                    ddp("nan in ys for precinct",p)
                    print(ys[p,:,:])
                    if include_nuisance:
                        print(logit_samps[p])
                    #
                    ddp("ns",ns[p])
                    ddp("vs",vs[p])
                    ddp("ecstar",ecstar)
                    ddp("ercstar",ercstar)
        pyro.sample("y", dist.Delta(ys).to_event(2))

        if include_nuisance:
            logit_samp_tensor = torch.cat(logit_samps,0)
            if not torch.all(torch.isfinite(logit_samp_tensor)):
                ddp("logits!!")
                for p in range(P):
                    if not torch.all(torch.isfinite(logit_samp_tensor[p,:,:])):
                        ddp("nan in nus for precinct",p)
                        print(logit_samp_tensor[p])
                        print(ysamps[p])
                        ddp("ns",ns[p])
                        ddp("vs",vs[p])
                        ddp("ecstar",ecstar)
                        ddp("ercstar",ercstar)
            pyro.sample("logits", dist.Delta(logit_samp_tensor).to_event(2))


    ##################################################################
    # ensure gradients get to the right place
    ##################################################################

    intermediate_vars = [
                pi,
                    Q,
                    ystars,
                    wstars,
                    wstars2,

                    ystars2 ,


                        logresidual_raw,
                            eprcstars ,
                        eprcstars2 ,g_delta,
                        ws,
                        ys,
                        logit_samp_tensor]
    [eachvar.retain_grad() for eachvar in
        intermediate_vars
    ]




    def fix_ec_grad():
        #dp"fixgrad",1)
        if torch.any(torch.isnan(ecstar_raw.grad)) or torch.any(torch.isnan(ec2r.grad)):
            ddp("ecstar_raw.grad",sizes(ecstar_raw.grad,ec2r.grad))
            #dp("2sp",[type(iv) for iv in intermediate_vars])
            ddp("2s",sizes(*[(iv.grad if iv.grad is not None else torch.tensor([])) for iv in intermediate_vars]))
            ddp("scoper",sizes(tots,vs,ns,indeps,ecstar_raw ,
                ercstar_raw))
            dat =[data,big_arrow,big_grad]
            import pdb; pdb.set_trace()
        ecstar_raw.grad = ecstar_raw.grad + ec2r.grad
        #dp("mode_star.grad",mode_star.grad)
    ecstar_raw.fix_grad = fix_ec_grad

    def fix_erc_grad():
        #dp"fixgrad",2)
        ercstar_raw.grad = ercstar_raw.grad + erc2r.grad
        if torch.any(torch.isnan(ercstar_raw.grad)) or torch.any(torch.isnan(erc2r.grad)):
            ddp("ercstar_raw.grad",sizes(ercstar_raw.grad,erc2r.grad))
            #dp("2s2p",[type(iv) for iv in intermediate_vars])
            ddp("2s2",sizes(*[(iv.grad if iv.grad is not None else torch.tensor([])) for iv in intermediate_vars]))

            #import pdb; pdb.set_trace()
        #dp("mode_star.grad",mode_star.grad)
    ercstar_raw.fix_grad = fix_erc_grad

    #no sdprc2 fix_grad. That's deliberate; the use of sdprc is an ugly hack and that error shouldn't bias the estimate of sdprc



    ##################################################################
    # Return values, debugging, cleanup.
    ##################################################################

    if DEBUG_ARROWHEAD: #debug arrowheads
        mini_indices = torch.tensor(range(2))
        mini_data = EISubData(data,mini_indices)
        mini_transformed_star_data = OrderedDict()
        for k,v in chain(gamma_star_data.items(),fstar_data.items()):
            if k in transformation:
                mini_transformed_star_data[k] = transformation[k](v)
            else:
                mini_transformed_star_data[k] = v

        for k,v in pstar_data.items():
            if k in transformation:
                mini_transformed_star_data[k] = transformation[k](v).index_select(0,mini_indices)
            else:
                mini_transformed_star_data[k] = v.index_select(0,mini_indices)

        mini_hess_center = pyro.condition(model,mini_transformed_star_data)
        #dp("line ",lineno())
        mini_mytrace = poutine.block(poutine.trace(mini_hess_center).get_trace)(mini_data, scale, include_nuisance)
        #dp("line ",lineno(),P,R,C)
        mini_log_posterior = mini_mytrace.log_prob_sum()+log_jacobian_adjustment
        ddp("mini_lp: ",mini_log_posterior,lineno())

        mini_y = pstar_data["y"].index_select(0,mini_indices)
        mini_y_delta = torch.zeros_like(mini_y)
        mini_y_delta[0,0,0] = 10
        mini_y_delta.requires_grad_(True)
        mini_y2 = mini_y + mini_y_delta
        mini_transformed_star_data["y"]=mini_y2
        mini_hess_center2 = pyro.condition(model,mini_transformed_star_data)
        #dp("line ",lineno())
        mini_mytrace2 = poutine.block(poutine.trace(mini_hess_center2).get_trace)(mini_data, scale,#527./2.,
                include_nuisance)
        #dp("line ",lineno(),P,R,C)
        mini_log_posterior2 = mini_mytrace2.log_prob_sum()+log_jacobian_adjustment
        ddp("mini_lp2: ",mini_log_posterior2,lineno())
        mini_h = myhessian.hessian(mini_log_posterior2,mini_y2)
        print(mini_h)
        print(myhessian.hessian(mini_log_posterior2,mini_y_delta))
        print(mini_y2[0])
        print(mini_y[0])
        import pdb; pdb.set_trace()

    if do_print:
        go_or_nogo.printstuff2()

    ddp("guide:end")

    if go_or_nogo.BREAK_NOW:
        import pdb; pdb.set_trace()
    result = dict(
        aa_gamma_star_data = gamma_star_data,
        fstar_data = fstar_data,
        pstar_data = pstar_data,
        all_means = all_means,
        big_arrow = big_arrow,
        big_grad = big_grad,
        adjusted_means = adjusted_means
    )
    return result










































data = pandas.read_csv('input_data/NC_precincts_2016_with_sample.csv')
  #,county,precinct,white_reg,black_reg,other_reg,test
wreg = torch.tensor(data.white_reg)
breg = torch.tensor(data.black_reg)
oreg = torch.tensor(data.other_reg)

def makePrecinctID(*args):
    return ":".join(args)

precinct_unique = [makePrecinctID(county,precinct) for (county,precinct) in zip(data.county,data.precinct)]

fixed_reg = [r.type(TTYPE) + FAKE_VOTERS_PER_RACE for r in [wreg, breg, oreg]]
ns = torch.stack(fixed_reg,1).type(TTYPE)


NCparams = EIData.load("NC_Data/NC_2016_statewide_alpha_and_beta.csv")
DUMMY_DATA = EIData(ns,None,precinct_unique,
        alpha=NCparams.alpha, beta=NCparams.beta)


def nameWithParams(filebase, data, S=None):
    if S is None:
        filename = (f"{filebase}_N{data.U}.csv")
    else:
        filename = (f"{filebase}_N{data.U}_S{S}.csv")
    return filename

def createOrLoadScenario(dummy_data = DUMMY_DATA,
            filebase="eiresults/"):
    filename = nameWithParams(filebase + "scenario", dummy_data)
    try:
        data = EIData.load(filename)
        print(filename, "from file")
    except IOError as e:
        print("exception:",e)
        assert not (os.path.exists(filename)) #don't just blindly overwrite!
        data = model(dummy_data)
        data.save(filename)
        print(filename, "created")
    return data

def saveFit(fitted_model_info, data, subsample_n, nparticles,nsteps,filebase="eiresults/funnyname_"):
    i = 0
    while True:
        filename = nameWithParams(filebase+"fit_"+str(i)+"_parts"+str(nparticles)+"_steps"+str(nsteps),
                data,subsample_n)
        if not os.path.exists(filename):
            break
        print("file exists:",filename)
        i += 1
    with open(filename, "w") as output:
        output.write(jsonize(fitted_model_info))
    #print("saveFit", filename)

def good_inits(noise=0.):
    ec,erc = legible_values(3,3)
    inits = dict(
            logsdrcstar=torch.log(erc.std()),
            logsdprcstar=torch.log(torch.tensor(SIM_SIGMA_NU)),
            ecstar_raw=ec[:2],
            ercstar_raw=erc[:2,:2],
            #globalpsi=,
            #precinctpsi=,
            )
    #TODO:noise
    return inits


def trainGuide(subsample_n = SUBSET_SIZE,
            filebase = "eiresults/",
            nparticles=1,inits=dict()): #honestly if I use nparticles, I should rewrite model&guide to both be over multiple samples. But including here now because it's easy.
    resetDebugCounts()

    # Let's generate voting data. I'm conditioning concentration=1 for numerical stability.

    data = createOrLoadScenario(DUMMY_DATA,filebase)
    N = data.U
    scale = N / subsample_n
    #dp(data)


    # Now let's train the guide.
    svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO(nparticles))

    pyro.clear_param_store()
    losses = []
    mean_losses = [] #(moving average)


    cur_perm = torch.randperm(N)
    used_up = 0

    for i in range(NSTEPS):

        if subsample_n < N:
            if (used_up + subsample_n) > N:
                cur_perm = torch.randperm(N)
                used_up = 0
            indices = cur_perm[used_up:used_up + subsample_n]
            used_up = used_up + subsample_n
        else:
            indices = torch.tensor(range(N))
        subset =  EISubData(data,indices)
        ddp("svi.step(...",i,scale,subset.indeps.size())
        loss = svi.step(subset,scale,True,do_print=(i % 10 == 0),
                inits=inits)
        if len(losses)==0:
            mean_losses.append(loss)
        else:
            mean_losses.append((
                                    mean_losses[-1] * (EXP_RUNNING_MEAN_WINDOW - 1)
                                    + min(
                                        loss,
                                        mean_losses[-1]*math.sqrt(EXP_RUNNING_MEAN_WINDOW)))
                                / EXP_RUNNING_MEAN_WINDOW)
        losses.append(loss)
        if i % 10 == 0:
            reload(go_or_nogo)
            go_or_nogo.printstuff(i,loss,mean_losses)
            curparams = pyro.get_param_store()
            print(f'epoch {i} loss = {loss:.2E}, mean_loss={mean_losses[-1]:.2E};'+
                f' sds = {dict(rc=float(torch.exp(curparams["logsdrcstar"])))};') #",prc=float(curparams["logsdprcstar"]))};')
            print(f' logitstar = {expand_and_center(curparams["ercstar_raw"])+expand_and_center(curparams["ecstar_raw"])}')
            if go_or_nogo.go:
                pass
            else:
                break
            try:
                if ((mean_losses[-1] > mean_losses[-CUTOFF_WINDOW])
                        and (mean_losses[-1] > mean_losses[-CUTOFF_WINDOW//2])):
                    ddp("Cutoff reached",mean_losses[-1], mean_losses[-CUTOFF_WINDOW])
                    break
            except Exception as e:
                pass

    ##

    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    if QUICKIE_SAVE:
        dataToSave = subset
    else:
        dataToSave = data
    fitted_model_info = guide(dataToSave, 1., True)
    fitted_model_info.update(
                    aacomment = "(add manually later)",
                    aaversion = EI_VERSION,
                    aaelasticity= dict(
                        MAX_NEWTON_STEP = MAX_NEWTON_STEP, #currently, just taking this much of a step, hard-coded
                        STARPOINT_AS_PORTION_OF_NU_ESTIMATE = STARPOINT_AS_PORTION_OF_NU_ESTIMATE,
                        NEW_DETACHED_FRACTION = NEW_DETACHED_FRACTION, #as in Newton, get it?
                        SDS_TO_REDUCE_BY = SDS_TO_REDUCE_BY,
                        SDS_TO_SHRINK_BY = SDS_TO_SHRINK_BY
                    ),
                    mean_loss = mean_losses[-1],
                    final_loss = loss
                    )

    saveFit(fitted_model_info, dataToSave, subsample_n, nparticles,i,filebase=filebase)
    ##

    for (key, val) in sorted(pyro.get_param_store().items()):
        if sum(val.size()) > 1:
            print(f"{key}:\n{val[:10]} (10 elems)")
        else:
            print(f"{key}:\n{val}")
    return(svi,losses,data)
