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
from collections import OrderedDict, defaultdict, Mapping


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

SEED = 4783019862  #Gingles2
torch.manual_seed(SEED)

pyro.enable_validation(True)
pyro.set_rng_seed(0)


EI_VERSION = "5.0.0"
FILEBASE = "ei_scaled_results/"
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

MAX_NEWTON_STEP = 1. #neutral = 1
STARPOINT_AS_PORTION_OF_NU_ESTIMATE = 1. #neutral=1
NEW_DETACHED_FRACTION = 0. #as in Newton, get it? neutral=0
SDS_TO_SHRINK_BY = 2/3 #neutral = 1.

REATTACH_GRAD_PORTION = 1. #neutral = 1.
SIGMA_NU_DETACHED_FRACTION = 1. #Neutral = 0. but very defensible up to 1. Moreover, seems to work!

NSTEPS = 2000
SUBSET_SIZE = 30
#BIG_PRIME = 73 #Wow, that's big!

FAKE_VOTERS_PER_RACE = 1.
FAKE_VOTERS_PER_REAL_PARTY = .5 #remainder go into nonvoting party

BASE_PSI = .01

QUICKIE_SAVE = (NSTEPS < 20) #save subset; faster
CUTOFF_WINDOW = 140
MIN_CUTOFF = 180
EXP_RUNNING_MEAN_WINDOW = 75


SIM_SIGMA_NU = .15

PSEUDOVOTERS_PER_CELL = 1.

DEBUG_ARROWHEAD = False

SDC = SDRC = 2.


SDPRC_STD = 1.2
SDPRC_MEAN = -2.5

if QUICKIE_SAVE:
    NUM_Y_SAMPS = 40
    SAVE_CHUNK_SIZE = 66
else:
    NUM_Y_SAMPS = 400
    SAVE_CHUNK_SIZE = 666


ICKY_SIGMA = False
PIN_PSI = True

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
    PSEUDOVOTERS_FOR_QS_VVAR = .5
    def __init__(self,ns,vs,
                    ids = None, ys=None, nus=None,
                    alpha = None, beta =None, sigmanu =None):# , sigmabeta=None):
        self.ns = toTypeOrNone(ns)
        self.vs = toTypeOrNone(vs)
        self.ys = toTypeOrNone(ys)
        self.nus = toTypeOrNone(nus)
        self.alpha = toTypeOrNone(alpha)
        self.beta = toTypeOrNone(beta)
        self.sigmanu = toTypeOrNone(sigmanu)
        try:
            if ids is None:
                self.ids=list(range(self.U))
            else:
                assert len(ids)==self.U
                self.ids = ids
        except:
            self.ids = None

    @reify
    def sqns(self):
        return torch.sqrt(self.ns)

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

    @reify #NOTE: code duplicated in EISubData
    def qs(self):
        return self.ys / self.ns.unsqueeze(-1)

    @reify #NOTE: code duplicated in EISubData
    def qs_vvar(self):
        vs_rbyc = self.vs.unsqueeze(1)+self.PSEUDOVOTERS_FOR_QS_VVAR*self.R
        ns_rbyc = self.ns.unsqueeze(-1)+self.PSEUDOVOTERS_FOR_QS_VVAR*self.C
        ys = self.ys+self.PSEUDOVOTERS_FOR_QS_VVAR
        return (self.ys * (vs_rbyc - self.ys) / vs_rbyc) / (ns_rbyc**2)

    @reify #NOTE: code duplicated in EISubData
    def qs_nvar(self):
        ns_rbyc = self.ns.unsqueeze(-1)+self.PSEUDOVOTERS_FOR_QS_VVAR*self.C
        ys = self.ys+self.PSEUDOVOTERS_FOR_QS_VVAR
        return (self.ys * (ns_rbyc - self.ys) / ns_rbyc) / (ns_rbyc**2)

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
            #writer.writedatarow(u"sigmabeta",self.sigmabeta)
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
            ys = nus = alpha = beta = sigmanu = ids = None
            for line in reader:
                var, u, r, c, val, id = line
                u, r, c = [int(a) for a in [u or 0,r or 0,c or 0]]
                if var==u"R":
                    #ddp("R",type(val),val)
                    R = int(val)
                    continue
                if var==u"C":
                    #ddp("C",type(val),val)
                    C = int(val)
                    continue
                if var==u"U":
                    #ddp("U",type(val),val)
                    U = int(val)
                    continue
                if var==u"sigmanu":
                    sigmanu = float(val)
                    continue
                if var==u"sigmabeta":
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
                pass
                #print("bad file load",e)
        #ddp("alpha",alpha)
        #ddp("beta",beta)
        data = cls(ns, vs, ids, ys, nus,
                    alpha, beta, sigmanu)
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
    sqns = sub(sqns=1)
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
    alpha = sub(alpha=0)
    beta = sub(beta=0)
    sigmanu = sub(sigmanu=0)

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

def model(data=None, scale=1., do_print=False, nsamps = 1, *args, **kwargs):
    """

    Notes:
        -if data is None, creates 30 precincts of data from nothing
        -otherwise, data should be a tuple (ns, vs, indeps, tots).
            -if vs is None, create & return data from model, with that ns
            -otherwise, model conditions on that vs (ie, use for density, not sampling)
    """
    for isamp in range(nsamps):
        if nsamps==1:
            iter = ""
        else:
            iter = f"{isamp:3}"
        P, R, C, ns, vs = data.getStuff

        prepare_ps = range(P)
        ps_plate = pyro.plate(f'{iter}all_sampled_ps',P)
        @contextlib.contextmanager
        def all_sampled_ps():
            with ps_plate as p, poutine.scale(scale=scale) as pscale:
                yield p

        sdc = SDC
        sdrc = SDRC #pyro.sample(f'{iter}sdrc', dist.LogNormal(SDRC_MEAN,SDRC_VAR))
        #if include_nuisance:
        sdprc = pyro.sample(f'{iter}sdprc', dist.LogNormal(SDPRC_MEAN,SDPRC_STD))

        if vs is None:
            if data.sigmanu is None:
                sdprc = SIM_SIGMA_NU
            else:
                sdprc = data.sigmanu
        #dp(f"sdprc in model:{sdprc}")

        ec = pyro.sample(f'{iter}ec', dist.Normal(torch.zeros(C),sdc).to_event(1))
        erc = pyro.sample(f'{iter}erc', dist.Normal(torch.zeros(R,C),sdrc).to_event(2))
        if vs is None:
            if data.alpha is None:
                ec,erc = legible_values(R,C)
            else:
                ec,erc = data.alpha, data.beta
                ddp("alpha beta",data.alpha, data.beta)


        #if include_nuisance:
        logit_centers = (ec + erc).expand(P,R,C)
        pi_star_raw = torch.exp(ec+erc)
        pi_star = pi_star_raw / torch.sum(pi_star_raw, 1, keepdim=True)
        sq_E_y_cond_on_gamma = data.sqns.unsqueeze(-1) * torch.sqrt(pi_star).expand(P,R,C)
        with all_sampled_ps() as p_tensor:
            logits = (
                pyro.sample(f'{iter}logits',
                    dist.Normal((ec + erc).expand(P,R,C), sdprc*sq_E_y_cond_on_gamma).to_event(2))
                ) #eprc.size() == [P,R,C] because plate dimension happens on left
        #dp("Model: sampling eprc",eprc[0,0,0])
        #else:
        #    logits = torch.zeros(P,R,C) #dummy for print statements. TODO:remove


        # with pyro.plate(f'{iter}candidatesm', C):
        #     ec = pyro.sample(f'{iter}ec', dist.Normal(0,sdc))
        #     with pyro.plate(f'{iter}rgroupsm', R):
        #         erc = pyro.sample(f'{iter}erc', dist.Normal(0,sdrc))
        #         if include_nuisance:
        #             with pyro.plate(f'{iter}precinctsm', P):
        #                 eprc = pyro.sample(f'{iter}eprc', dist.Normal(0,sdprc))

        with all_sampled_ps() as p_tensor:#pyro.plate(f'{iter}precinctsm2', P):
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
                y = pyro.sample(f"{iter}y",
                            CMult(1000,logits=logits).to_event(1))
                            #dim P, R, C from plate, to_event, CMult
                            #note that n is totally fake — sums are what matter.
                            #TODO: fix CMult so this fakery isn't necessary.
                #dp("sampled y",y[0])
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

            return EIData(ns,vs,data.ids,y, logits - ec - erc, ec, erc, sdprc)

    ddp("model:end")#, sizes(ns,vs,ec,erc,logits,y))



def expand_and_center(tens, return_ldaj=False, ignore_dims=[]):
    result = tens
    for i,n in enumerate(tens.size()):
        if i not in ignore_dims:
            result = torch.cat([result,
                        -torch.sum(result,i).unsqueeze(i)]
                    ,i)

    if return_ldaj:
        return (result,0.) #Not actually zero, but constant, so whatevs.
    return result


def get_param(inits,name,default,as_constant=False,*args,**kwargs):
    if as_constant:
        return inits.get(name,default).clone().detach().requires_grad_(True)
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
        return (result,-torch.sum(t))
    return result

def softmax(t,minval=0.,mult=80.):
    if minval == 0.:
        mins = torch.zeros_like(t)
    else:
        mins = torch.ones_like(t)*minval*mult
    return torch.logsumexp(torch.stack((t*mult,mins)),0)/mult


def guide(data, scale, do_print=False, inits=dict(), nsamps = 1, icky_sigma=ICKY_SIGMA,
            *args, **kwargs):
    ddp("guide:begin",scale)


    ##################################################################
    # Set up plates / weights
    ##################################################################
    ns, vs, indeps, tots = data.ns, data.vs, data.indeps, data.tots
    P = len(ns)
    R = len(ns[1])
    C = len(vs[1])


    ##################################################################
    # Get guide parameters (stars)
    ##################################################################

    gamma_star_data = OrderedDict()
    fstar_data = OrderedDict() #"frequentist", empirical bayes, ick... whatever you want to call it
    pstar_data = OrderedDict()
    transformation = defaultdict(lambda: lambda x: x) #factory of identity functions

    #logsdrcstar = get_param(inits,'logsdrcstar',ts(0.))
    #fstar_data.update(sdrc=logsdrcstar)
    #transformation.update(sdrc=exp_ldaj)
    #if include_nuisance:
    #logsdprcstar = get_param(inits,'logsdprcstar',ts(-3.))
    #fstar_data.update(sdprc=logsdprcstar)
    #transformation.update(sdprc=exp_ldaj)
    eprcstar_startingpoint = torch.zeros(P,R,C,requires_grad =True) #not a pyro param...
    #eprcstar_startingpoint[p].requires_grad_(True) #...so we have to do this manually

    ec_then_erc_star = get_param(inits,'ec_then_erc_star', torch.zeros(R,C-1))

    ecstar_raw = ec_then_erc_star[0,:]#get_param(inits,'ecstar_raw', torch.zeros(C-1))
    ercstar_raw = ec_then_erc_star[1:,:]#get_param(inits,'ercstar_raw', torch.zeros(R-1,C-1))
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

    ecerc2r = ec_then_erc_star.clone().detach().requires_grad_()
    ec2r = ecerc2r[0,:]
    erc2r = ecerc2r[1:,:]#ercstar_raw.clone().detach().requires_grad_()

    ec2 = expand_and_center(ec2r)
    erc2 = expand_and_center(erc2r)
    #dp("sizes",sizes(ec2r,erc2r,ec2,erc2))


    #Including expand_and_center makes erc not identifiable; this is an issue for writeup, not coding


    #Amortize stars

    logittotals = ec2 + erc2
    #dp("sizes1 ",P,R,C,eprc.size(),ec.size(),erc.size(),logittotals.size())
    #if include_nuisance:
    #dp("adding starting point")
    logittotals = logittotals + eprcstar_startingpoint # += doesn't work here because mumble in-place mumble shape
    #else:
    #    logittotals = logittotals.expand(P,R,C)
    pi_raw = torch.exp(logittotals)
    pi = pi_raw / torch.sum(pi_raw,-1, keepdim=True)

    sq_E_y_cond_on_gamma = data.sqns.unsqueeze(-1) * torch.sqrt(pi).expand(P,R,C)



    log_jacobian_adjustment_hessian = torch.tensor(0.)
    log_jacobian_adjustment_elbo_global = torch.tensor(0.)

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
        log_jacobian_adjustment_hessian += ldaj


        #get ν̂^(0)
        #if include_nuisance:
        QbyR = Q/torch.sum(Q,-1).unsqueeze(-1)
        logresidual_raw = torch.log(QbyR / pi) / sq_E_y_cond_on_gamma
        nsUbyRby1 = torch.sum(ystars,2,keepdim=True)
        lr_var_of_like = (nsUbyRby1-ystars)/(ystars*nsUbyRby1)/sq_E_y_cond_on_gamma**2
            #1 sd down, rescaled, logged, minus orig; rough estimate of sd of likelihood of logresidual

        #sign_residual = logresidual_raw.sign()
        #abs_residual = logresidual_raw * sign_residual
        #shrunk_residual = softmax(abs_residual - lr_sd_of_like * SDS_TO_REDUCE_BY) * sign_residual
        sdprc_raw = torch.sqrt(torch.mean(softmax(logresidual_raw**2-  lr_var_of_like)))
        if SIGMA_NU_DETACHED_FRACTION == 1.:

            sdprc = sdprc_raw.clone().detach().requires_grad_(True) #* SIGMA_NU_DETACHED_FRACTION + sdprc_raw * (1.-SIGMA_NU_DETACHED_FRACTION)
        else:
            sdprc = sdprc_raw.clone().detach() * SIGMA_NU_DETACHED_FRACTION + sdprc_raw * (1.-SIGMA_NU_DETACHED_FRACTION)

        if icky_sigma:
            fstar_data.update(sdprc=sdprc)
        else:
            lsdprc = torch.log(sdprc)
            gamma_star_data.update(sdprc=lsdprc)
            transformation.update(sdprc=exp_ldaj)
        #lr_prec_of_like = lr_var_of_like ** -2 #sd to precision

        eprcstars = (sq_E_y_cond_on_gamma *
                STARPOINT_AS_PORTION_OF_NU_ESTIMATE* logresidual_raw/
                    lr_var_of_like/SDS_TO_SHRINK_BY/
                (1/lr_var_of_like/SDS_TO_SHRINK_BY + 1/sdprc**2))
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
    #if include_nuisance:
    logits = expand_and_center(ecstar_raw) + expand_and_center(ercstar_raw) + eprcstars2
    #else:

    #    logits = (expand_and_center(ecstar_raw) + expand_and_center(ercstar_raw)).repeat(P,1,1)
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
            log_jacobian_adjustment_elbo_global += ldaj
            log_jacobian_adjustment_hessian += ldaj
        else:
            transformed_star_data[k] = v

    #
    #dp("line ",lineno())
    hess_center = pyro.condition(model,transformed_star_data)
    #dp("line ",lineno())
    mytrace = poutine.block(poutine.trace(hess_center).get_trace)(data, scale)
    #dp("line ",lineno(),P,R,C)
    log_posterior = mytrace.log_prob_sum()+log_jacobian_adjustment_hessian
    ddp("lp: ",log_posterior,log_jacobian_adjustment_hessian,lineno())



    hessian_stars_in_sampling_order = [] #this will have eprcstars — good for taking hessian but wrong for mean
    mean_stars_in_sampling_order = [] #this will have logits — right mean, but not upstream, so can't use for hessian
    for part_name in list(gamma_star_data.keys()):
        #dp(f"adding {part_name} to stars_in_sampling_order")
        hessian_stars_in_sampling_order.append(gamma_star_data[part_name])
        mean_stars_in_sampling_order.append(gamma_star_data[part_name])
    gamma_dims = sum(gamma_part.numel() for gamma_part in gamma_star_data.values())
    #dp("len gamma",len(stars_in_sampling_order),stars_in_sampling_order)

    #add pstar_data to stars_in_sampling_order — but don't get it from pstar_data because it comes in wrong format
    #if include_nuisance:
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
    #else:
    #    tensors_per_unit = 1 #tensors, not elements=(R-1)*(C-1)
    #    dims_per_unit = (R-1)*(C-1)
    #    hessian_stars_in_sampling_order.extend(wstars_list) #`wstars_list` is already a list, not a tensor, so this works
    #    mean_stars_in_sampling_order.extend(wstars_list) #`wstars_list` is already a list, not a tensor, so this works
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
    # Psis
    ##################################################################
    #declare global-level psi params
    globalpsi = get_param(inits,'globalpsi',torch.ones(gamma_dims)*BASE_PSI,
                constraint=constraints.positive,
                as_constant=PIN_PSI)
    #declare precinct-level psi params
    precinctpsi = get_param(inits,'precinctpsi',BASE_PSI * torch.ones(dims_per_unit),
                constraint=constraints.positive,
                as_constant=PIN_PSI)

    #dp"setpsis",sizes(globalpsi,precinctpsi))
    big_arrow.setpsis(globalpsi,precinctpsi)
    big_arrow.weights = [scale] * P


    ##################################################################
    # Put Jacobian into guide density
    ##################################################################
    #stupid_global = torch.exp(log_jacobian_adjustment_elbo_global)
    junk = pyro.sample("jacobian",
                        dist.Delta(torch.zeros(1),
                                    log_density=-log_jacobian_adjustment_elbo_global),#, stupid_global),
                        infer={'is_auxiliary': True}) #there's probably a better way but whatevs.

    ##################################################################
    # Loop (multiple particles)
    ##################################################################
    gamma_chol = torch.cholesky(big_arrow.marginal_gg_cov())
    lambda_chols = [torch.cholesky(big_arrow.llinvs[p]) for p in range(P)]
    for isamp in range(nsamps):
        log_jacobian_adjustment_elbo_local = torch.zeros(1)
        if nsamps==1:
            iter = ""
        else:
            iter = f"{isamp:3}"

        ##################################################################
        # "Empirical bayes" parameters — just optimize, don't get distribution
        ##################################################################
        for k,v in fstar_data.items():
            pyro.sample(iter+k, dist.Delta(transformation[k](v)))

        prepare_ps = range(P) #for dealing with stared quantities (no pyro.sample)
        ps_plate = pyro.plate(f'{iter}all_sampled_ps',P)
        @contextlib.contextmanager
        def all_sampled_ps(): #for dealing with unstared quantities (include pyro.sample)
            with ps_plate as p, poutine.scale(scale=scale) as pscale:
                yield p

        ##################################################################
        # Sample gamma (globals)
        ##################################################################



        #head_precision, head_adjustment,  = rescaledSDD(-neg_big_hessian,combinedpsi) #TODO: in-place


        #gamma_info = big_hessian[:gamma_dims,:gamma_dims]

        all_means = torch.cat([tpart.contiguous().view(-1) for tpart in mean_stars_in_sampling_order],0)
        #dp("all_means",all_means)
        #dp(torch.any(torch.isnan(torch.diag(neg_big_hessian))),torch.any(torch.isnan(torch.diag(big_hessian))))
        gamma_mean = all_means[:gamma_dims]
        #dp("detirminants",np.linalg.det(gamma_info.clone().detach()),np.linalg.det(big_hessian.clone().detach()))
        #dp(gamma_info[:3,:3])
        #dp(-neg_big_hessian[:6,:3])

        gamma = pyro.sample(f'{iter}gamma',
                        dist.OMTMultivariateNormal(gamma_mean, gamma_chol),
                        infer={'is_auxiliary': True})
        g_delta = gamma - gamma_mean

        #decompose gamma into specific values
        tmpgamma = gamma
        for pname, pstar in gamma_star_data.items():
            elems = pstar.nelement()
            pdat, tmpgamma = tmpgamma[:elems], tmpgamma[elems:]
            #dp(f"adding {pname} from gamma ({elems}, {pstar.size()}, {tmpgamma.size()}, {pdat})" )

            if pname in transformation:
                pyro.sample(iter+pname, dist.Delta(transformation[pname](pdat.view(pstar.size())))
                                    .to_event(len(list(pstar.size())))) #TODO: reshape after transformation, not just before???
            else:
                pyro.sample(iter+pname, dist.Delta(pdat.view(pstar.size()))
                                    .to_event(len(list(pstar.size()))))
        assert list(tmpgamma.size())[0] == 0





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

            #precinct_cov = big_arrow.llinvs[p] #for Newton's method, not for sampling #TODO: This is actually the same as conditional_cov; remove?

            precinct_grad = big_grad.index_select(0,precinct_indices) #[gamma_dims + pp*(R-1)*(C-1): gamma_dims + (pp+1)*(R-1)*(C-1)]

            #dp("precinct:::",gamma_1p_hess.size(),precinct_cov.size(),big_grad.size(),precinct_grad.size(),)
            #if include_nuisance:
            adjusted_mean_raw = conditional_mean + step_mult * torch.mv(big_arrow.llinvs[p], precinct_grad)
            adjusted_mean = adjusted_mean_raw.clone().detach() * NEW_DETACHED_FRACTION + adjusted_mean_raw * (1 - NEW_DETACHED_FRACTION)
                                 #one (partial, as defined by step_mult) step of Newton's method
                                 #Note: this applies to both ws and nus (eprcs). I was worried about whether that was circular logic but I talked with Mira and we both think it's actually principled.
            #else:
            #    adjusted_mean = conditional_mean


            adjusted_means.append(adjusted_mean)

            try:
                pstuff = pyro.sample(f"{iter}pstuff_{p}",
                                dist.OMTMultivariateNormal(adjusted_mean, lambda_chols[p]),
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

            #if include_nuisance:
            logit = pstuff[(R-1)*(C-1):].view(1,R,C)
            logit_samps.append(logit)

        with all_sampled_ps():
            ws = torch.stack(wsamps)
            ys,ldaj = polytopizeU(R,C,ws,indeps,return_ldaj=True)
            log_jacobian_adjustment_elbo_local += ldaj
            if not torch.all(torch.isfinite(ys)):
                for p in range(P):
                    if not torch.all(torch.isfinite(ys[p,:,:])):
                        ddp("nan in ys for precinct",p)
                        print(ys[p,:,:])
                        #if include_nuisance:
                        print(logit_samps[p])
                        #
                        ddp("ns",ns[p])
                        ddp("vs",vs[p])
                        ddp("ecstar",ecstar)
                        ddp("ercstar",ercstar)
            pyro.sample(f"{iter}y", dist.Delta(ys).to_event(2))

            #if include_nuisance:
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
            pyro.sample(f"{iter}logits", dist.Delta(logit_samp_tensor).to_event(2))


        ##################################################################
        # Put local Jacobian into guide density
        ##################################################################
        #stupid_local = torch.exp(log_jacobian_adjustment_elbo_local)
        #if stupid_local>0 and torch.isfinite(stupid_local):
        junk = pyro.sample(f"{iter}jacobian",
                    dist.Delta(torch.zeros(1),# stupid_local),
                            log_density=-log_jacobian_adjustment_elbo_local),
                    infer={'is_auxiliary': True}) #there's probably a better way but whatevs.

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




    def fix_ecerc_grad():
        #dp"fixgrad",1)
        if torch.any(torch.isnan(ec_then_erc_star.grad)) or torch.any(torch.isnan(ecerc2r.grad)):
            ddp("ecstar_raw.grad",sizes(ec_then_erc_star.grad,ecerc2r.grad))
            #dp("2sp",[type(iv) for iv in intermediate_vars])
            ddp("2s",sizes(*[(iv.grad if iv.grad is not None else torch.tensor([])) for iv in intermediate_vars]))
            ddp("scoper",sizes(tots,vs,ns,indeps,ecstar_raw ,
                ercstar_raw))
            dat =[data,big_arrow,big_grad]
            import pdb; pdb.set_trace()
        ec_then_erc_star.grad = ec_then_erc_star.grad + ecerc2r.grad * REATTACH_GRAD_PORTION
        #dp("mode_star.grad",mode_star.grad)
    ec_then_erc_star.fix_grad = fix_ecerc_grad


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
        mini_mytrace = poutine.block(poutine.trace(mini_hess_center).get_trace)(mini_data, scale)
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
            )
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
        adjusted_means = adjusted_means,
        apsis = dict(globalpsi = globalpsi,
                    precinctpsi = precinctpsi),
        weight=scale,
        log_posterior=log_posterior
    )
    return result









































if QUICKIE_SAVE:
    data = pandas.read_csv('input_data/NC_precincts_2016_with_sample.csv')
else:
    data = pandas.read_csv('input_data\ALL_precincts_2016_reg_with_sample_60.csv')

  #,county,precinct,white_reg,black_reg,other_reg,test
wreg = torch.tensor(data.white_reg)
breg = torch.tensor(data.black_reg)
oreg = torch.tensor(data.other_reg)

def makePrecinctID(*args):
    return ":".join(str(args))

precinct_unique = [makePrecinctID(county,precinct) for (county,precinct) in zip(data.county,data.precinct)]

fixed_reg = [r.type(TTYPE) + FAKE_VOTERS_PER_RACE for r in [wreg, breg, oreg]]
ns = torch.stack(fixed_reg,1).type(TTYPE)


NCparams = EIData.load("NC_Data/NC_2016_statewide_alpha_and_beta.csv")
DUMMY_DATA = EIData(ns,None,precinct_unique,
        alpha=NCparams.alpha, beta=NCparams.beta)


def nameWithParams(filebase, data, dversion="", S=None, extension =None, N=None):
    if N is None:
        N = data.U
    if extension is None:
        extension = ".json" if S else ".csv"
    if S is None:
        filename = (f"{filebase}_SIG{data.sigmanu}_{dversion}_N{N}{extension}")
    else:
        filename = (f"{filebase}_SIG{data.sigmanu}_{dversion}_N{N}_S{S}_{extension}")
    return filename

def createOrLoadScenario(dummy_data = DUMMY_DATA, dversion="",
            filebase=FILEBASE):
    filename = nameWithParams(filebase + "scenario", dummy_data, dversion)
    try:
        data = EIData.load(filename)
        print(filename, "from file")
        return data
    except IOError as e:
        print("exception:",e)
    assert not (os.path.exists(filename)) #don't just blindly overwrite!
    data = model(dummy_data)
    data.save(filename)
    print(filename, "created")
    return data

def saveFit(fitted_model_info, data, subsample_n, nparticles,nsteps,dversion="",filebase=FILEBASE + "funnyname_",i=None):

    filename = nameWithParams(filebase+"fit_"+str(i)+"_parts"+str(nparticles)+"_steps"+str(nsteps),
            data,dversion,subsample_n)
    if i is None:
        i = 0
        while True:
            filename = nameWithParams(filebase+"fit_"+str(i)+"_parts"+str(nparticles)+"_steps"+str(nsteps),
                    data,dversion,subsample_n)
            if not os.path.exists(filename):
                break
            print("file exists:",filename)
            i += 1
    with open(filename, "a") as output:
        output.write(jsonize(fitted_model_info))
    #print("saveFit", filename)
    return i

def good_inits(shrink=1.,sd=False,noise=0.):
    #ec,erc = legible_values(3,3)
    ec,erc = (NCparams.alpha, NCparams.beta)
    ec_then_erc_star = torch.cat([ec[:2].view(1,2),erc[:2,:2]],0)
    inits = dict(
            ec_then_erc_star=ec_then_erc_star * shrink
            #ecstar_raw=ec[:2],
            #ercstar_raw=erc[:2,:2],
            #globalpsi=,
            #precinctpsi=,
            )
    if sd:
        pass
        #inits.update(
        #        logsdrcstar=torch.log(erc.std())
        #        )
    #TODO:noise
    return inits

def base_logits_of(gammas,R,C,wdim):
    alphams = gammas[:,:C-1]
    alphas = expand_and_center(alphams,ignore_dims=[0])
    betams = gammas[:,C-1:C-1+wdim]
    betas = expand_and_center(betams.view(-1,R-1,C-1),ignore_dims=[0])
    return alphas.unsqueeze(1) + betas

def model_density_of(ys,nus,base_logits,sigma_nu,R,C,wdim):
    logits = base_logits + nus.view(-1,R,C)
    denses = torch.zeros(ys.size()[0])

    for r in range(R):
        modr = CMult(logits=logits[:,r])
        denses += modr.log_prob(ys[:,r])
    nuprobs = torch.distributions.Normal(0,sigma_nu.unsqueeze(1)).log_prob(nus)
    nps = torch.sum(nuprobs,1)
    dp("np complete?",sizes(nps,ys,nus,base_logits,sigma_nu,nuprobs,denses))
    #import pdb; pdb.set_trace()
    #denses += nps
    return denses,nps


def weighted_incremental_variance(dataWeightPairs):
    #never used, but pasted here for reference
    wSum = wSum2 = mean = S = 0

    for x, w in dataWeightPairs:  # Alternatively "for x, w in zip(data, weights):"
        wSum = wSum + w
        #wSum2 = wSum2 + w*w
        meanOld = mean
        mean = meanOld + (w / wSum) * (x - meanOld)
        S = S + w * (x - meanOld) * (x - mean)

    population_variance = S / wSum
    # Bessel's correction for weighted samples
    # Frequency weights
    #sample_frequency_variance = S / (wSum - 1)
    # Reliability weights
    #sample_reliability_variance = S / (wSum - wSum2/wSum)




    # aa_gamma_star_data = gamma_star_data,
    # fstar_data = fstar_data,
    # pstar_data = pstar_data,
    # all_means = all_means,
    # big_arrow = big_arrow,
    #         G = self.G,
    #         L = self.L,
    #         gg_raw = self.gg,
    #         raw_lls = self.raw_lls,
    #         gg_cov = self.gg_cov,
    #         chol_lls = self.chol_lls,
    #         llinvs = self.llinvs,
    #         gls = self.gls,
    #         weights = self.weights
    # big_grad = big_grad,
    # adjusted_means = adjusted_means
def sampleYs(fit,data,n,previousSamps = None,weightToUndo=1.,indices=None, icky_sigma=ICKY_SIGMA):
    ba = fit["big_arrow"]
    am = fit["all_means"]
    G,L = ba.G, ba.L
    R, C = (data.R,data.C)
    wdim = (R-1)*(C-1)

    if previousSamps is None:
        YSums = torch.zeros(n,R,C)
        ldajsums = torch.zeros(n)
        guidesampdenses = torch.zeros(n)
        moddenses = torch.zeros(n)
        modnps = torch.zeros(n)

        dgamma = torch.distributions.MultivariateNormal(am[:G], ba.marginal_gg_cov())
        gammas = dgamma.sample([n])
        base_logits = base_logits_of(gammas,R,C,wdim)
        guidesampdenses += dgamma.log_prob(gammas)

        Qvarcounters = torch.zeros(n,R,C,3)
                #cols correspond to: 0:wSum,
                                    #1:mean,
                                    #2:S
        #wSumsQ, wSum2Q, meanQ, SQ = [torch.zeros(n) for _i in range(4)]



    else:
        gammas,YSums, stuff, Qvarcounters = previousSamps
        guidesampdenses,ldajsums,moddenses,modnps = [stuff[:,i] for i in range(4)]
        dp("sampleYs0",sizes(gammas,YSums, stuff,guidesampdenses,ldajsums,moddenses,modnps))
        #import pdb; pdb.set_trace()
        base_logits = base_logits_of(gammas,R,C,wdim)


    ll = ba.llinvs
    U = len(ll)
    if indices is None:
        indices = range(U)
    for u in indices:
        lstar = am[G+u*L:G+u*L+L]#wdim]
        ignore_nu = False
        if ignore_nu:
            raise Exception("unimplemented")
            #wmean = wstar.unsqueeze(0) + torch.matmul(gammas.unsqueeze(1), ba.gls[u][:,:wdim] )
            #dw = torch.distributions.MultivariateNormal(wmean, ll[u][:wdim,:wdim].unsqueeze(0))
        else:
            lmean = lstar.unsqueeze(0) + torch.matmul(gammas.unsqueeze(1), ba.gls[u])
            dl = torch.distributions.MultivariateNormal(lmean, ll[u].unsqueeze(0) * weightToUndo)
        lambdas = dl.sample()
        llp = dl.log_prob(lambdas).squeeze()
        lambdas = lambdas.squeeze()

        guidesampdenses += llp
        ws = lambdas[:,:wdim].view(n,R-1,C-1)
        #import pdb; pdb.set_trace()
        ys,ldajs = polytopizeU(R, C, ws, data.indeps[u:u+1].expand(n,R*C), return_ldaj=True, return_plural=True)
        if icky_sigma:
            sigma_nu = fit["fstar_data"]["sdprc"].unsqueeze(0)
            dp("sdprc",sizes(sigma_nu))
        else:
            sigma_nu = torch.exp(gammas[:,-1])
        denses, nps = model_density_of(ys,lambdas[:,wdim:],base_logits,sigma_nu,R,C,wdim)
        moddenses = moddenses + denses
        modnps = modnps + nps
        dp("sampleYs", sizes(lambdas,lmean,lstar,ll,gammas,base_logits,ldajs))
        ldajsums = ldajsums + ldajs.squeeze()
        YSums = YSums + ys

        nrbyc = data.ns[u].view(1,R,1) #used as weights for running variance calculation
        qs = ys / nrbyc
        #"w" = nrbyc ; "x" = qs
        Qvarcounters[:,:,:,0] = Qvarcounters[:,:,:,0] + nrbyc
        #Qvarcounters[:,:,:,69] = Qvarcounters[:,:,:,69] + nrbyc**2
        meanOld = Qvarcounters[:,:,:,1].clone()
        Qvarcounters[:,:,:,1] = meanOld + (nrbyc / Qvarcounters[:,:,:,0]) * (qs - meanOld)
        Qvarcounters[:,:,:,2] = Qvarcounters[:,:,:,1] + nrbyc * (qs - meanOld) * (qs - Qvarcounters[:,:,:,1])

    denses = torch.stack([guidesampdenses,ldajsums,moddenses,modnps],1)
    dp("denses",sizes(denses))
    return [t.clone().detach().requires_grad_(False) for t in (gammas,YSums, denses,Qvarcounters)]

def saveYsamps(samps, data, subsample_n, nparticles,nsteps,dversion="",filebase=FILEBASE + "funnyname_",i=None,N=None):

    filename = nameWithParams(filebase+"dsamps_"+str(i)+"_parts"+str(nparticles)+"_steps"+str(nsteps),
            data,dversion,subsample_n,extension=".csv",N=N)
    if i is None:
        i = 0
        while True:
            filename = nameWithParams(filebase+"dsamps_"+str(i)+"_parts"+str(nparticles)+"_steps"+str(nsteps),
                    data,dversion,subsample_n,extension=".csv",N=N)
            if not os.path.exists(filename):
                break
            print("file exists:",filename)
            i += 1
    with open(filename, "a") as output:
        writer = csv.writer(output)
        #import pdb; pdb.set_trace()
        for samp in zip(*samps):
            writer.writerow([float(val) for atensor in samp for val in atensor.view(-1)])
    return i

def detachRecursive(obj):
    if torch.is_tensor(obj):
        return obj.clone().detach().requires_grad_(False)
    if isinstance(obj, Mapping):
        return dict((k,detachRecursive(v)) for k,v in obj.items())
    if type(obj) is list:
        return [detachRecursive(it) for it in obj]
    if isinstance(obj, ArrowheadPrecision):
        for attr in ("G", "L", "gg", "gls", "raw_lls", "weights", "lls", "_mgg",
                    "psig", "psil","vecweights","vecraw_lls","chol_lls","llinvs",
                    "gg_cov","vecgls","_mgg_chol" ): #Too many??
            if hasattr(obj,attr):
                setattr(obj,attr,detachRecursive(getattr(obj,attr)))
        return obj
    if isinstance(obj, EIData):

                # self.ns = toTypeOrNone(ns)
                # self.vs = toTypeOrNone(vs)
                # self.ys = toTypeOrNone(ys)
                # self.nus = toTypeOrNone(nus)
                # self.alpha = toTypeOrNone(alpha)
                # self.beta = toTypeOrNone(beta)
                # self.sigmanu = toTypeOrNone(sigmanu)
        for attr in ("ns", "vs", "ys", "nus", "alpha", "beta", "sigmanu" ): #Too many??
            if hasattr(obj,attr):
                setattr(obj,attr,detachRecursive(getattr(obj,attr)))
        return obj
    return obj

def rerunGuide(data,guide,mean_losses,loss,subsample_n, nsamps,dversion,filebase,num_y_samps,steps,stride=SAVE_CHUNK_SIZE):
    U = data.U
    savedSoFar = 0
    numChunks = U//stride
    if numChunks < 3:
        numChunks = numChunks + 1
    stride = 1 + U//numChunks #balance them out
    ifit = None
    isamps = None
    print("rerunGuide",U,savedSoFar,isamps,stride)
    Ysamps = None

    cur_perm = torch.randperm(U)
    while savedSoFar < U:
        print("    rerunGuide",U,savedSoFar,isamps,stride)
        indices = cur_perm[savedSoFar:min(savedSoFar+stride,U)]
        savedSoFar = savedSoFar+stride
        dataToSave = EISubData(data,indices)
        dataSize = dataToSave.U
        weight = U/dataSize
        fitted_model_info = guide(dataToSave, weight)
        fitted_model_info = detachRecursive(fitted_model_info)
        fitted_model_info.update(
                        aacomment = "(add manually later)",
                        aaversion = EI_VERSION,
                        aaelasticity= dict(
                            MAX_NEWTON_STEP = MAX_NEWTON_STEP, #currently, just taking this much of a step, hard-coded
                            STARPOINT_AS_PORTION_OF_NU_ESTIMATE = STARPOINT_AS_PORTION_OF_NU_ESTIMATE,
                            NEW_DETACHED_FRACTION = NEW_DETACHED_FRACTION, #as in Newton, get it?
                            SDS_TO_SHRINK_BY = SDS_TO_SHRINK_BY,
                            REATTACH_GRAD_PORTION = REATTACH_GRAD_PORTION,
                            SIGMA_NU_DETACHED_FRACTION = SIGMA_NU_DETACHED_FRACTION, #Neutral = 0. but very defensible up to 1.
                            ICKY_SIGMA = ICKY_SIGMA,
                            PIN_PSI = PIN_PSI,
                            SEED = SEED
                        ),
                        mean_loss = mean_losses[-1],
                        final_loss = loss
                        )

        ifit  = saveFit(fitted_model_info, dataToSave, subsample_n, nsamps,steps,dversion=dversion,filebase=filebase,i=ifit)

        Ysamps = sampleYs(fitted_model_info, dataToSave, num_y_samps, Ysamps, weight)
        del fitted_model_info

    isamps = saveYsamps(Ysamps, dataToSave, subsample_n, nsamps,steps,dversion=dversion,filebase=filebase,i=isamps,N=U)



def trainGuide(subsample_n = SUBSET_SIZE,
            filebase = FILEBASE,
            nsteps=NSTEPS,
            sigmanu = SIM_SIGMA_NU,
            dummydata = DUMMY_DATA,
            nsamps=1,dversion="",inits=dict(),
            num_y_samps=NUM_Y_SAMPS,
            force_full=False):
    resetDebugCounts()

    # Let's generate voting data. I'm conditioning concentration=1 for numerical stability.

    dummydata.sigmanu = sigmanu
    data = createOrLoadScenario(dummydata,filebase=filebase,dversion=dversion)
    N = data.U
    scale = N / subsample_n
    #dp(data)


    # Now let's train the guide.
    svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO(1))

    pyro.clear_param_store()
    losses = []
    mean_losses = [] #(moving average)


    cur_perm = torch.randperm(N)
    used_up = 0

    for i in range(nsteps):

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
        loss = svi.step(subset,scale,do_print=(i % 10 == 0),nsamps=nsamps,
                inits=inits)
        loss = detachRecursive(loss) #I hate memory leaks!
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
            print(f'epoch {i} loss = {loss:.2E}, mean_loss={mean_losses[-1]:.2E};') #",prc=float(curparams["logsdprcstar"]))};')
            print(f' logitstar = {expand_and_center(curparams["ec_then_erc_star"][1:,:])+expand_and_center(curparams["ec_then_erc_star"][0,:])}')
            if go_or_nogo.go:
                pass
            else:
                break
            try:
                if ((mean_losses[-1] > mean_losses[-CUTOFF_WINDOW])
                        and (mean_losses[-1] > mean_losses[-CUTOFF_WINDOW//2])):
                    if i > MIN_CUTOFF:
                        ddp("Cutoff reached",mean_losses[-1], mean_losses[-CUTOFF_WINDOW])
                        break
            except Exception as e:
                pass

    ##

    try:
        plt.plot([min(loss,mean_losses[-1]*5) for loss in losses])
        plt.xlabel('epoch')
        plt.ylabel('loss')
    except Exception as e:
        print("Problem plotting:",e)

    print("trainGuide post..................................................")
    for ii in range(3):
        print(",,")

    for (key, val) in sorted(pyro.get_param_store().items()):
        if sum(val.size()) > 1:
            print(f"{key}:\n{val[:10]} (10 elems)")
        else:
            print(f"{key}:\n{val}")

    if (not force_full) and (QUICKIE_SAVE or nsteps < 30):
        dataToSave = subset
    else:
        dataToSave = data

    for ii in range(2):
        print("::")

    rerunGuide(dataToSave,guide,mean_losses,loss,subsample_n, nsamps,dversion,filebase,num_y_samps,i)

    print("Done trainGuide..................................................")
    for ii in range(10):
        print(".")





    return(svi,losses,data)

def modelQvar(samps=30,
        sigmanu = SIM_SIGMA_NU,
        dummydata = DUMMY_DATA,
        filebase=FILEBASE):

    dummydata.sigmanu = sigmanu
    print("runs")
    runs = [detachRecursive(model(DUMMY_DATA)) for _i in range(samps)]
    print("qses")
    qses = torch.stack([data.qs for data in runs],0) #samps x U x R x C
    print("vvars")
    vvars = torch.stack([data.qs_vvar for data in runs],0) #samps x U x R x C
    print("vvarmeans")
    vvarmeans = vvars.mean(1) #samps x R x C
    print("nvars")
    nvars = torch.stack([data.qs_nvar for data in runs],0) #samps x U x R x C
    print("vvarmeans")
    nvarmeans = nvars.mean(1) #samps x R x C
    print("raw qvars")
    qvars = qses.var(1) #samps x R x C
    print("qvars_corrected")
    qvars_corrected = (softmax(qvars-vvarmeans)) #samps x R x C
    result = (qvars_corrected, qvars, vvarmeans, nvarmeans)
    print("mQv sizes",sizes(qses,vvars,vvarmeans,qvars,qvars_corrected))
    saveYsamps(result, runs[0], "XX", "XX","XX",dversion="0",filebase=filebase+"modelQvar",i="XX",N=dummydata.U)
    return result
