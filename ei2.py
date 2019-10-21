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
from utilities.rank1torch_vectorized import optimize_Q_objectly
from utilities import go_or_nogo
from utilities.cmult import CMult
from utilities import polytopize
reload(polytopize)
from utilities.polytopize import get_indep, polytopize, depolytopize, to_subspace, process_data
from utilities.posdef import *

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

ts = torch.tensor


torch.manual_seed(478301986) #Gingles

pyro.enable_validation(True)
pyro.set_rng_seed(0)



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

ADJUST_SCALE = .05
MAX_NEWTON_STEP = .7 #currently, just taking this much of a step, hard-coded
EPRCstar_HESSIAN_POINT_FRACTION = .5
RECENTER_PRIOR_STRENGTH = 2.



NSTEPS = 5001
SUBSET_SIZE = 5
BIG_PRIME = 73 #Wow, that's big!

FAKE_VOTERS_PER_RACE = 1
FAKE_VOTERS_PER_REAL_PARTY = .5 #remainder go into nonvoting party

MINIMAL_DEBUG = False
if MINIMAL_DEBUG:
    pyrosample = lambda x,y,infer=None : y.sample()
else:
    pyrosample = pyro.sample

class EIData:
    def __init__(self,ns,vs):
        self.ns = ns
        self.vs = vs

    @reify
    def tots(self):
        return torch.sum(ns,1)

    @reify
    def indeps(self):
        assert approx_eq(self.tots,torch.sum(self.vs,1)), f'#print("sums",{self.tots},{torch.sum(self.vs,1)})'
        return torch.matmul(torch.unsqueeze(self.nns,-1),torch.unsqueeze(self.nvs,-2)) / self.tots.unsqueeze(-1).unsqueeze(-1)

    @reify
    def nns(self): #normalized ns
        return self.ns/self.tots.unsqueeze(1)


    @reify
    def nvs(self): #normalized ns
        return self.vs/torch.self.tots.unsqueeze(1)

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

def sub(**kwargs): #I really hate typing quotation markes
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



def model(data=None, scale=1., include_nuisance=True, do_print=False):
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
    sdrc = pyrosample('sdrc', dist.Exponential(.2))
    if include_nuisance:
        sdprc = pyrosample('sdprc', dist.Exponential(.2))

    if vs is None:
        sdc = .2
        sdrc = .4
        sdprc = .6
    #dp(f"sdprc in model:{sdprc}")

    #This is NOT used for data, but instead as a way to sneak a "prior" into the guide to improve identification.
    #param_residual=pyrosample('param_residual', dist.Normal(0.,1.))

    ec = pyrosample('ec', dist.Normal(torch.zeros(C),sdc).to_event(1))
    erc = pyrosample('erc', dist.Normal(torch.zeros(R,C),sdrc).to_event(2))
    if include_nuisance:
        with all_sampled_ps() as p_tensor:
            logits = (
                pyrosample(f'logits', dist.Normal((ec + erc).expand(P,R,C), sdprc).to_event(2))
                ) #eprc.size() == [P,R,C] because plate dimension happens on left
        #dp("Model: sampling eprc",eprc[0,0,0])
    else:
        logits = torch.zeros(P,R,C) #dummy for print statements. TODO:remove

    if vs is None:
        erc = torch.zeros([R,C])
        erc[0] = ts(range(C))
        erc[1,0] = ts(2.)
        ec = torch.zeros(C)
        ec[1] = .5

    # with pyro.plate('candidatesm', C):
    #     ec = pyrosample('ec', dist.Normal(0,sdc))
    #     with pyro.plate('rgroupsm', R):
    #         erc = pyrosample('erc', dist.Normal(0,sdrc))
    #         if include_nuisance:
    #             with pyro.plate('precinctsm', P):
    #                 eprc = pyrosample('eprc', dist.Normal(0,sdprc))

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
                        dp("modsize2",p,r,[a for a in [ns[p,r], logits[p,r]]])
                        raise
        else:
            if not torch.all(torch.isfinite(logits)):
                dp("logits!!!")
                for p in range(P):
                    if not torch.all(torch.isfinite(logits[p])):
                        dp("nan in logits[p]",p)
                        dp(logits[p])
                        dp(ec)
                        dp(erc)
            y = pyro.sample(f"y",
                        CMult(1000,logits=logits).to_event(1))
                        #dim P, R, C from plate, to_event, CMult
                        #note that n is totally fake — sums are what matter.
                        #TODO: fix CMult so this fakery isn't necessary.
            dp("model y",scale,y.size(),y[0,:2,:2])
            #dp("ldim",logits.size(),y[0,0,0])

    if vs is None:
        #
        dp(f"ec:{ec}")
        dp(f"erc:{erc}")
        dp(f"y[0]:{y[0]}")
        vs = torch.sum(y,1)

        return EIData(ns,vs)

    dp("model:end")



def expand_and_center(tens):
    result = tens
    for i,n in enumerate(tens.size()):
        result = torch.cat([result,
                    -torch.sum(result,i).unsqueeze(i)]
                ,i)

    return result




# This function is to guess a good starting point for eprcstar. But removing for now because:
# 1. SIMPLIFY!
# 2. We're doing 1 (partial) step of Newton's Method anyway so this is mostly redundant.


# def initial_eprc_star_guess(totsp,pi_p,Q2,Q_precision,pi_precision):
#     pi_precision = totsp / pi_p / (torch.ones_like(pi_p) - pi_p)
#     pweighted_Q = pi_p + (Q2-pi_p) * Q_precision / (Q_precision + pi_precision)
#     return (torch.log(pweighted_Q/pi_p))

def guide(data, scale, include_nuisance=True, do_print=False):
    dp("guide:begin",scale,include_nuisance)


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

    logsdrcstar = pyro.param('logsdrcstar',ts(-1.))
    fstar_data.update(sdrc=logsdrcstar)
    transformation.update(sdrc=torch.exp)
    if include_nuisance:
        logsdprcstar = pyro.param('logsdprcstar',ts(-1.))
        fstar_data.update(sdprc=logsdprcstar)
        transformation.update(sdprc=torch.exp)
        eprcstar_startingpoint = torch.zeros(P,R,C,requires_grad =True) #not a pyro param...
        #eprcstar_startingpoint[p].requires_grad_(True) #...so we have to do this manually

    ecstar_raw = pyro.param('ecstar_raw', torch.zeros(C-1))
    ercstar_raw = pyro.param('ercstar_raw', torch.zeros(R-1,C-1))
    gamma_star_data.update(ec=ecstar_raw,erc=ercstar_raw)
    transformation.update(ec=expand_and_center, erc=expand_and_center)

    eprcstar_hessian_point_fraction = EPRCstar_HESSIAN_POINT_FRACTION #pyro.param("eprcstar_fraction",ts(0.5),constraint=constraints.interval(0.0, 1.0))
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
    dp("sizes",[it.size() for it in [ec2r,erc2r,ec2,erc2,]])

    if include_nuisance:
        sdprc2 = logsdprcstar.detach().requires_grad_()

    #Including expand_and_center makes erc not identifiable
    #We could  "impose a prior" as follows, but instead, we'll just allow things to wander off.
    # recentering_amount = (RECENTER_PRIOR_STRENGTH *
    #         torch.mean((ercstar - expand_and_center(ercstar))**2)/torch.exp(logsdrcstar))
    # pyrosample('param_residual', dist.Delta(recentering_amount))


    #Amortize stars
    ystar = []
    wstar = [] #P  * ((R-1) * (C-1)
    ystar2 = [] #reconstituted as a function of wstar
    eprcstars = [] #P  * (R * C)

    logittotals = ec2 + erc2
    #dp("sizes1 ",P,R,C,eprc.size(),ec.size(),erc.size(),logittotals.size())
    if include_nuisance:
        #dp("adding starting point")
        logittotals = logittotals + eprcstar_startingpoint # += doesn't work here because mumble in-place mumble shape
    else:
        logittotals = logittotals.expand(P,R,C)
    pi_raw = torch.exp(logittotals)
    pi = pi_raw / torch.sum(pi_raw,-1).unsqueeze(-1)



    #dp("guide:pre-p")

    for p in prepare_ps:#pyro.plate('precinctsg2', P):
        #precalculation - logits to pi


        dp("amosize",[a.size() for a in [pi,normvs, normns]])

        Q, iters = optimize_Q_objectly(pi,data,tolerance=.01,maxiters=3)

        #get ŷ^(0
        #dp(f"optimize_Q_objectly {p}:{iters}")
        ystar.append(Q*tots[p])
        if p==0:
            pass
            #dp("p0", Q,tots[p])

        #depolytopize
        wstar.append(depolytopize(R,C,ystar[p],indeps[p]))

        ystar2.append(polytopize(R,C,wstar[p],indeps[p]))

        QbyR = Q/torch.sum(Q,-1).unsqueeze(-1)
        logresidual = torch.log(QbyR / pi[p])
        eprcstar = logresidual * eprcstar_hessian_point_fraction

        #get ν̂^(0)
        if include_nuisance:
            eprcstars.append(eprcstar)
            #was: initial_eprc_star_guess(tots[p],pi[p],Q2,Q_precision,pi_precision))


    pstar_data.update(y=torch.cat([y.unsqueeze(0) for y in ystar2],0))
    #dp("y is",pstar_data["y"].size(),ystar[-1].size(),ystar[0][0,0],ystar2[0][0,0])
    if include_nuisance:
        eprcstar = torch.cat([eprc.unsqueeze(0) for eprc in eprcstars],0)
        logits = expand_and_center(ecstar_raw) + expand_and_center(ercstar_raw) + eprcstar
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
            transformed_star_data[k] = transformation[k](v)
        else:
            transformed_star_data[k] = v

    #
    #dp("line ",lineno())
    hess_center = pyro.condition(model,transformed_star_data)
    #dp("line ",lineno())
    mytrace = poutine.block(poutine.trace(hess_center).get_trace)(data, scale, include_nuisance)
    #dp("line ",lineno(),P,R,C)
    log_posterior = mytrace.log_prob_sum()
    dp("lp: ",log_posterior,lineno())



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
        dims_per_unit = (R-1)*(C-1) + R*C
        for w,eprc,logit in zip(wstar,
                        eprcstars, #This is NOT the center of the distribution; tstar's `logits`
                                #However, the Hessian wrt `eprcstars` should equal that wrt `logits`
                                #And using `logits` here wouldn't work because it's not upstream on the autodiff graph
                        logits):
            hessian_stars_in_sampling_order.extend([w,eprc])
            mean_stars_in_sampling_order.extend([w,logit])
    else:
        tensors_per_unit = 1 #tensors, not elements=(R-1)*(C-1)
        dims_per_unit = (R-1)*(C-1)
        hessian_stars_in_sampling_order.extend(wstar) #`wstar` is already a list, not a tensor, so this works
        mean_stars_in_sampling_order.extend(wstar) #`wstar` is already a list, not a tensor, so this works
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
    # Sample gamma (globals)
    ##################################################################

    #declare global-level psi params
    globalpsi = pyro.param('globalpsi',torch.ones(gamma_dims)*BASE_PSI,
                constraint=constraints.positive)
    #declare precinct-level psi params
    precinctpsi = pyro.param('precinctpsi',BASE_PSI * torch.ones(dims_per_unit),
                constraint=constraints.positive)

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

    gamma = pyrosample('gamma',
                    dist.MultivariateNormal(gamma_mean, precision_matrix=big_arrow.marginal_gg()),
                    infer={'is_auxiliary': True})
    g_delta = gamma - gamma_mean

    #decompose gamma into specific values
    tmpgamma = gamma
    for pname, pstar in gamma_star_data.items():
        elems = pstar.nelement()
        pdat, tmpgamma = tmpgamma[:elems], tmpgamma[elems:]
        #dp(f"adding {pname} from gamma ({elems}, {pstar.size()}, {tmpgamma.size()}, {pdat})" )

        if pname in transformation:
            pyrosample(pname, dist.Delta(transformation[pname](pdat.view(pstar.size())))
                                .to_event(len(list(pstar.size())))) #TODO: reshape after transformation, not just before???
        else:
            pyrosample(pname, dist.Delta(pdat.view(pstar.size()))
                                .to_event(len(list(pstar.size()))))
    assert list(tmpgamma.size())[0] == 0


    for k,v in fstar_data.items():
        pyrosample(k, dist.Delta(transformation[k](v)))



    ##################################################################
    # Sample lambda_i (locals)
    ##################################################################

    #TODO: uncomment the following, which is fancy logic for learning what fraction of a Newton's method step to take

    #precinct_newton_step_multiplier_logit = pyro.param(
    #        'precinct_newton_step_multiplier_logit',ts(0.))
    #epnsml = torch.exp(precinct_newton_step_multiplier_logit)
    step_mult = MAX_NEWTON_STEP #* epnsml / (1 + epnsml)
    ysamps = []
    logit_samps = []
    adjusted_means = []
    global_indices = ts(range(gamma_dims))
    for p in range(P):


        precinct_indices = ts(range(gamma_dims + p*dims_per_unit, gamma_dims + (p+1)*dims_per_unit))

        #dp(f"gamma_1p_hess:{P},{p},{B},{P//B},{len(big_HWs)},{big_HWs[p//B].size()},")
        #dp(f"HW2:{big_HWs[p//B].size()},{list(full_indices)}")
        conditional_mean, conditional_cov = big_arrow.conditional_ll_mcov(p,g_delta,
                                all_means.index_select(0,precinct_indices))

        precinct_cov = big_arrow.llinvs[p] #for Newton's method, not for sampling

        precinct_grad = big_grad.index_select(0,precinct_indices) #[gamma_dims + pp*(R-1)*(C-1): gamma_dims + (pp+1)*(R-1)*(C-1)]

        #dp("precinct:::",gamma_1p_hess.size(),precinct_cov.size(),big_grad.size(),precinct_grad.size(),)
        if include_nuisance:
            adjusted_mean = conditional_mean + step_mult * torch.mv(precinct_cov, precinct_grad)
                                 #one (partial, as defined by step_mult) step of Newton's method
                                 #Note: this applies to both ws and nus (eprcs). I was worried about whether that was circular logic but I talked with Mira and we both think it's actually principled.
        else:
            adjusted_mean = conditional_mean


        adjusted_means.append(adjusted_mean)

        try:
            pstuff = pyrosample(f"pstuff_{p}",
                            dist.MultivariateNormal(adjusted_mean, conditional_cov),
                            infer={'is_auxiliary': True})

        except:
            dp("error sampling pstuff",p,conditional_cov.size())
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
        y = polytopize(R,C,w_raw.view(R-1,C-1),indeps[p])
        ysamps.append(y.view(1,R,C))

        if include_nuisance:
            logit = pstuff[(R-1)*(C-1):].view(1,R,C)
            logit_samps.append(logit)

    with all_sampled_ps():
        ys = torch.cat(ysamps,0)
        if not torch.all(torch.isfinite(ys)):
            for p in range(P):
                if not torch.all(torch.isfinite(ys[p,:,:])):
                    dp("nan in ys for precinct",p)
                    print(ys[p,:,:])
                    if include_nuisance:
                        print(logit_samps[p])
                    #
                    dp("ns",ns[p])
                    dp("vs",vs[p])
                    dp("ecstar",ecstar)
                    dp("ercstar",ercstar)
        pyro.sample("y", dist.Delta(ys).to_event(2))

        if include_nuisance:
            logit_samp_tensor = torch.cat(logit_samps,0)
            if not torch.all(torch.isfinite(logit_samp_tensor)):
                dp("logits!!")
                for p in range(P):
                    if not torch.all(torch.isfinite(logit_samp_tensor[p,:,:])):
                        dp("nan in nus for precinct",p)
                        print(logit_samp_tensor[p])
                        print(ysamps[p])
                        dp("ns",ns[p])
                        dp("vs",vs[p])
                        dp("ecstar",ecstar)
                        dp("ercstar",ercstar)
            pyro.sample("logits", dist.Delta(logit_samp_tensor).to_event(2))


    ##################################################################
    # ensure gradients get to the right place
    ##################################################################

    def fix_ec_grad():
        ecstar_raw.grad = ecstar_raw.grad + ec2r.grad
        #dp("mode_star.grad",mode_star.grad)
    ecstar_raw.fix_grad = fix_ec_grad

    def fix_erc_grad():
        ercstar_raw.grad = ercstar_raw.grad + erc2r.grad
        #dp("mode_star.grad",mode_star.grad)
    ercstar_raw.fix_grad = fix_erc_grad

    #no sdprc2 fix_grad. That's deliberate; the use of sdprc is an ugly hack and that error shouldn't bias the estimate of sdprc



    ##################################################################
    # Return values, debugging, cleanup.
    ##################################################################


    if do_print:
        go_or_nogo.printstuff2()

    dp("guide:end")

    result = dict(
        aa_gamma_star_data = gamma_star_data,
        fstar_data = fstar_data,
        pstar_data = pstar_data,
        a_hess_center = hess_center,
        all_means = all_means,
        big_arrow = big_arrow,
        big_grad = big_grad,
        adjusted_means = adjusted_means
    )
    return result



def get_subset(data,size,i):

    ns, vs, indeps, tots = data
    P = data.U
    indices = ts([((i*size + j)* BIG_PRIME) % P for j in range(size)])
    scale = P / size #likelihood impact of a precinct proportional to n
    dp("scale",scale)
    return (EISubData(data,indices),scale)



data = pandas.read_csv('input_data/NC_precincts_2016_with_sample.csv')
  #,county,precinct,white_reg,black_reg,other_reg,test
wreg = torch.tensor(data.white_reg)
breg = torch.tensor(data.black_reg)
oreg = torch.tensor(data.other_reg)

fixed_reg = [r + FAKE_VOTERS_PER_RACE for r in [wreg, breg, oreg]]
ns = torch.stack(fixed_reg,1).double()
DUMMY_DATA = EIData(ns,None)


def trainGuide():
    resetDebugCounts()

    # Let's generate voting data. I'm conditioning concentration=1 for numerical stability.

    data = model(DUMMY_DATA)
    processed_data = process_data(data) #precalculate independence points
    #dp(data)


    # Now let's train the guide.
    svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO())

    pyro.clear_param_store()
    losses = []
    for i in range(NSTEPS):
        subset, scale = get_subset(processed_data,SUBSET_SIZE,i)
        dp("svi.step(...",i,scale)
        loss = svi.step(subset,scale,True,do_print=(i % 10 == 0))
        losses.append(loss)
        if i % 10 == 0:
            reload(go_or_nogo)
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
    # print("ystar[0]:",
    #     polytopize(4,3,pyroStore["wstar_0"],
    #                get_indep(4,3,ns[0],vs[0])))

    return(svi,losses,data)
