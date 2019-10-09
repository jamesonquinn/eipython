#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

from importlib import reload
import csv
import time
import math
import os
import random #Mixing seeds — not reproducible — TODO:fix
import itertools
import json
from collections import Mapping

from matplotlib import pyplot as plt
from collections import OrderedDict, defaultdict

import torch
from torch.distributions import constraints
import contextlib

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam, AdagradRMSProp
from pyro import poutine
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.infer.mcmc import NUTS
#from pyro.infer.mcmc.api import MCMC
import numpy as np
import pandas as pd

if True:#False:#
    from utilities import myhessian
else:
    import hessian as myhessian
from utilities.lambertw import lambertw
from utilities.polytopize import approx_eq
from utilities import go_or_nogo
from utilities.deltamvn import DeltaMVN
from utilities.posdef import *
#from utilities.stanCache import StanModel_cache
ts = torch.tensor

pyro.enable_validation(True)
pyro.set_rng_seed(0)



SAFETY_MULT = 1.01
LOG_BASE_PSI = math.log(BASE_PSI)

EULER_CONSTANT = 0.5772156649015328606065120900824024310421
GUMBEL_SD = math.pi/math.sqrt(6.)


BASE_COMPLAINTS_REMAINING = 10
BASE_YAYS_REMAINING = 10
COMPLAINTS_REMAINING = BASE_COMPLAINTS_REMAINING
YAYS_REMAINING = BASE_YAYS_REMAINING


DF_ADJ_QUANTUM = 0.02

N_SAMPLES = 400
SUBSAMPLE_N = 11

LAMBERT_MAX_ITERS = 10
LAMBERT_TOL = 1e-2

MAX_OPTIM_STEPS = 2001



MIN_DF = 2.5
SMEAN = 0. #ie, 1
SSCALE = 2.
DMEAN = 1. #ie, 2.7
DSCALE = 1.5
MIN_SIGMA_OVER_S = 1.9

FUCKING_TENSOR_TYPE = type(torch.tensor(1.))


EVIL_HACK_EPSILON = 0.00000001 #OMG this is evil



data = pd.read_csv('testresults/effects_errors.csv')
echs_x = torch.tensor(data.effects)
echs_errors = torch.tensor(data.errors)
N = len(echs_x)
base_scale = 1.
modal_effect = 1.*base_scale
tdom_fat_params = [dict(modal_effect=modal_effect,
                            df=3., #actual df is 3
                            t_scale=10.)] #actual sigma is 10
#
ndom_fat_params = [dict(modal_effect=modal_effect,
                            df=3., #actual df is 3
                            t_scale=2.)]#actual sigma is 2
#
tdom_norm_params = [dict(modal_effect=modal_effect,
                            df=30., #actual df is 30
                            t_scale=10.)]#actual sigma is 10
#
ndom_norm_params = [dict(modal_effect=modal_effect,
                            df=30., #actual df is 30
                            t_scale=2.)]#actual sigma is 2

dummy_echs_params = [dict(modal_effect=0.,
                            df=0., #actual df is 30
                            t_scale=0.)]#actual sigma is 2


def complain(*args):
    global COMPLAINTS_REMAINING
    COMPLAINTS_REMAINING -= 1
    if COMPLAINTS_REMAINING >= 0:
        print("complaint",COMPLAINTS_REMAINING,*args)
    if COMPLAINTS_REMAINING % 20 == 0:
        print("complaint",COMPLAINTS_REMAINING)


def yay(*args):
    global YAYS_REMAINING
    YAYS_REMAINING -= 3
    if YAYS_REMAINING >= 0:
        print("yay",YAYS_REMAINING,*args)
    if YAYS_REMAINING % 20 == 0:
        print("yay",YAYS_REMAINING)

def model(N,full_N,indices,x,full_x,errors,full_errors,maxError,
                        save_data=None,
                        weight=1.,
            smean=ts(SMEAN),sscale=ts(SSCALE),dmean = ts(DMEAN),dscale=ts(DSCALE),
            fixedParams = None): #groups, subgroups, groupsize by trial, options
    """
    Notes:
        For un-amortized guides, full_* is actually full data. For amortized
        guides, full_* is just a copy of the not-full version.
    """

    #print("model??", maxError,weight,scalehyper)

    units_plate = pyro.plate('units',N)
    @contextlib.contextmanager
    def chosen_units():
        with units_plate as n, poutine.scale(scale=weight) as nscale:
            yield n

    if N==full_N:
        #print("model fu",N)
        full_units = chosen_units
    else:
        #print("model fu",full_N)
        full_units = pyro.plate('full_units',full_N)


    #prior on μ_x
    modal_effect = pyro.sample('modal_effect',dist.Normal(ts(0.),ts(20.)))

    min_sigma = maxError*MIN_SIGMA_OVER_S
    #prior on sd(τ)
    t_scale = min_sigma   +pyro.sample('t_scale_raw',dist.LogNormal(smean,sscale))

    #prior on df
    df = MIN_DF + pyro.sample('dfraw',dist.LogNormal(dmean,dscale))

    if fixedParams is not None:
        print("generating data",fixedParams)
        #
        modal_effect = fixedParams['modal_effect']
        t_scale = min_sigma + torch.exp(fixedParams['t_scale'])
        df = (MIN_DF + torch.exp(fixedParams['df'])).requires_grad_()

    #print("model t_part",N,full_N)
    if N==full_N:
        with pyro.plate('2full_units',full_N): #I hate you! but this magic works? #chosen_units:#
            t_part = pyro.sample('t_part',dist.StudentT(df,torch.zeros(full_N),ts(1.) * t_scale))
    else:
        with pyro.plate('2full_units',N):
            with poutine.scale(scale=weight):#chosen_units:#full_units:
                t_part = pyro.sample('t_part',dist.StudentT(df,torch.zeros(N),ts(1.) * t_scale))


    #print("model t_part", t_part)

    #Latent true values (offset)
    truth = modal_effect + t_part

    #Observations conditional on truth (likelihood)
    if fixedParams is not None:
        print("creating",modal_effect.size(),t_part.size(),t_scale.size(),truth.size(),errors.size())
        observations = pyro.sample('observations', dist.Normal(truth,errors).to_event(1))
        print("Not none - creating data.")
    else:
        try:
            with chosen_units():
                observations = pyro.sample('observations', dist.Normal(truth,errors), obs=x)
        except:
            print("ERROR", truth, errors)
            print("ERROR", modal_effect, norm_scale, t_scale)
            raise

    if fixedParams is not None:
        return (t_part.detach(),observations.detach())
    #print("end model",modal_effect,norm_scale,t_scale,t_part)

def cuberoot(x):
    return torch.sign(x) * torch.abs(x) ** (1./3.)


def evil_hack_fix(t): #ensure a tensor has no zero elements by adding or subtracting epsilon
    #this is because the gradient of the cube root of 0 is NaN. Usually, the dimension where
    #the 0 occurs isn't even in the subsample but it still blows up the overall gradient. So just
    #force things not to be 0 and it works. Yuck.
    if torch.randint(0,2,[1]) == 0:
        evil_hack_epsilon = EVIL_HACK_EPSILON
    else:
        evil_hack_epsilon = -EVIL_HACK_EPSILON
    t2 = t.detach()
    return t + ((t2==0).float() * evil_hack_epsilon)

def getMLE(nscale, tscale, obs, df, return_intermediates=False):
    #assert getDiscriminant(nscale, tscale, obs, df) < 0 #only one root
    try:
        assert torch.all(tscale / nscale > MIN_SIGMA_OVER_S) #only one root, regardless of obs; redundant, stronger
    except:
        print("assertion scale fail",tscale, df)
        print(torch.max(nscale), torch.min(nscale), df)
        print(nscale)
        raise
    #assert False

    b = -obs
    tsq = tscale ** 2
    c = tsq * df + nscale ** 2 * (df + 1)
    d = -tsq * obs * df

    p = -b/3
    q = p**3 + (b*c-3*d)/6
    r = c/3

    inner = q**2 + (r-p**2)**3
    if torch.any(inner<0):
        raise Exception("imaginary case not implemented")
    else:
        insqrt = torch.sqrt(inner)

    plusterm, minusterm = (q + insqrt, q - insqrt)
    plustermfixed = evil_hack_fix(plusterm )
    minustermfixed = evil_hack_fix(minusterm)
    crpt,crmt = (cuberoot(plustermfixed), cuberoot(minustermfixed))
    x = crpt + crmt + p
    shouldBeZero = x**3 + b * x**2 + c * x + d

    try:
        assert approx_eq(shouldBeZero,torch.zeros(1))
    except:
        badIx = (torch.abs(shouldBeZero)>.01).nonzero()
        complain(
                "assert approx_eq:",badIx,
                "\n              1..",(x**3)[badIx], (b * x**2)[badIx], (c * x)[badIx], (d)[badIx],
                "\n              2..",shouldBeZero[badIx])
        # raise
    # print("getMLE(",nscale, tscale, obs, df)
    # print("getMLE2(",b, c, d)
    # print("getMLE3(",p, q, r, inner, insqrt, x)
    # print("getMLE3(", (q - insqrt), (q - insqrt) ** (1./3.))
    if return_intermediates:
        return (x,[b,c,d,p,q,r,inner,insqrt,plustermfixed, minustermfixed,crpt,crmt,shouldBeZero,tsq])
    return x

def getscaleddens(nscale,tscale,df,np,tp):
    #print("getdens(",nscale,tscale,df,np,tp)
    return (dist.Normal(0,nscale).log_prob(np) +
            dist.StudentT(df,0,tscale).log_prob(tp))

def getdens(nscale,tscale,df,np,tp):
    return getscaleddens(nscale,tscale,df,np,tp)

def testMLE(df=3.,n=20):
    for i in range(n):
        fac = -torch.log(torch.rand([1]))
        tscale = fac * (1. - .6 * torch.rand([1]))
        nscale = fac - torch.abs(tscale)
        obs = fac * (-torch.log(torch.rand([1]))-1)
        tpr = getMLE(nscale,tscale,obs,df)
        npr = obs - tpr
        print(obs,nscale,tscale,npr,tpr)
        ml = getdens(nscale,tscale,df,npr,tpr)
        for delta in [.0001,-.0001]:
            mlnot = getdens(nscale,tscale,df,(npr+delta),(tpr-delta))
            if mlnot < ml:
                print(f"getMLE seems to have failed, {fac} {nscale} {tscale} {obs} {npr} {tpr} {delta} {ml} {mlnot}")

                print(fac.item())
                plt.plot([getdens(nscale,tscale,df,(npr+delta),(tpr-delta)) for delta in np.linspace(-.5*fac.item(), .5*fac.item(), 100)])
                plt.xlabel('dg')
                plt.ylabel('dens')
                plt.show()
                plt.clf()
                #raise "Broken!"

def plotMLE():
    nscale = ts(.5)
    tscale = ts(1.)
    obs = ts(1.)
    df = ts(3.)
    facs = ts([float(i) for i in range(25,200)]) / 50.
    plt.plot([getMLE(nscale*fac,tscale,obs,df) for fac in facs])
    plt.xlabel('nscale')
    plt.ylabel('t')
    plt.show()
    plt.clf()

    plt.plot([getMLE(nscale,tscale*fac,obs,df) for fac in facs])
    plt.xlabel('tscale')
    plt.ylabel('t')
    plt.show()
    plt.clf()

    plt.plot([getMLE(nscale,tscale,obs*fac,df) for fac in facs])
    plt.xlabel('obs')
    plt.ylabel('t')
    plt.show()
    plt.clf()

def get_unconditional_cov(full_precision, n):
    #TODO: more efficient
    return(torch.inverse(full_precision)[:n,:n])

def laplace_guide(N,full_N,indices,x,full_x,errors,full_errors,maxError,
                        save_data=None,
                        weight=1.,scalehyper=ts(4.),tailhyper=ts(10.),
                        amortize=True):
    #print("laplace_guide:",N,len(x),len(errors),weight)


    units_plate = pyro.plate('units',N)
    @contextlib.contextmanager
    def chosen_units():
        with units_plate as n, poutine.scale(scale=1.) as nscale:#weight) as nscale:
            yield n

    if N==full_N:
        #print("guide fu",N)
        full_units = chosen_units
    else:
        #print("guide fu",full_N)
        @contextlib.contextmanager
        def full_units():
            with pyro.plate('full_units',full_N) as n:
                yield n

    hat_data = OrderedDict() #
    fhat_data = OrderedDict() #frequentist
    transformations = defaultdict(lambda: lambda x: x) #
    mode_hat = pyro.param("mode_hat",ts(0.))
    hat_data.update(modal_effect=mode_hat)

    ltscale_hat = pyro.param("ltscale_hat",ts(0.))
    hat_data.update(t_scale_raw=ltscale_hat)
    transformations.update(t_scale_raw=torch.exp)

    ldfraw_hat2 = pyro.param("ldfraw_hat",ts(0.))
    if save_data and "df_adj" in save_data:
        ldfraw_hat = ldfraw_hat2 + save_data["df_adj"]
    else:
        ldfraw_hat = ldfraw_hat2
    fhat_data.update(dfraw=ldfraw_hat)
    #hat_data.update(fhat_data) #include frequentist value in Hessian
        #can't, no Hessians of gamma functions
    transformations.update(dfraw=torch.exp)


    dm = mode_hat.detach().requires_grad_()
    dtr = ltscale_hat.detach().requires_grad_() #detached... raw
    ddfr = ldfraw_hat.detach().requires_grad_() #detached... raw

    dt =  maxError*MIN_SIGMA_OVER_S + torch.exp(dtr) #detached, cook
    ddf = MIN_DF + torch.exp(ddfr) #detached, cook


    #print("laplace_guide:",torch.max(errors),maxError,torch.max(errors)/dt)
    if amortize:
        (full_tpart,MLEvars) = getMLE(full_errors, dt, full_x-dm, ddf,
                                            return_intermediates=True)
    else:
        full_tpart = pyro.param("full_tmode", full_x - torch.ones(full_N) * torch.mean(full_x))

    if N == full_N:
        sub_tpart = full_tpart
    else:
        sub_tpart = full_tpart.index_select(0,indices)

    #print("tpart:",tpart)

    #true_g_hat.requires_grad = True
    gamma_names = list(hat_data.keys())
    #print("guide t_part",N,full_N,sub_tpart.size(),full_tpart.size())
    hat_data.update(t_part=sub_tpart)

    #Get hessian
    phi = torch.cat([phiPart.view(-1) for phiPart in hat_data.values()],0)

    conditioner = dict()
    conditioner.update((k,transformations[k](v)) for k, v in itertools.chain(hat_data.items(), fhat_data.items()))
    hessCenter = pyro.condition(model,conditioner)
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)(N,full_N,#None,x,x,errors,errors,
                                indices,x,full_x,errors,full_errors,
                                maxError,save_data,weight,scalehyper,tailhyper) #*args,**kwargs)
    logPosterior = blockedTrace.log_prob_sum()

    gamma_parts = dict((gamma_name,hat_data[gamma_name]) for gamma_name in gamma_names)
    #count parameters
    G = int(sum(gamma_parts[pname].nelement() for pname in gamma_names))

    if save_data and "df_adj" in save_data: #just running to get logPosterior; stop and return
        grad = myhessian.mygradient(logPosterior, hat_data.values())
        save_data.update(logPosterior=logPosterior, grad=grad)
        return(save_data)
    (hess, grad) = myhessian.arrowhead_hessian(logPosterior, hat_data.values(), headsize=len(gamma_names),
                                blocksize=1, return_grad=True)#, allow_unused=True)
    hess = -hess
    globalpsiraw = pyro.param("globalpsi",torch.zeros(G) + LOG_BASE_PSI)
    latentpsiraw = pyro.param("latentpsi",torch.zeros(1) + LOG_BASE_PSI)
    globalpsi = torch.exp(globalpsiraw)
    #ensure positive definite
    head_precision, head_adjustment, lowerblocks = getMpD(hess,G,1,globalpsi,torch.exp(latentpsiraw),weight)
    #big_precision = rescaledSDD(hess, torch.exp(lpsi), head=3, weight=weight)


    #invert matrix (maybe later, smart)
    gamma_cov = torch.inverse(head_precision)

    #MVN = dist.MultivariateNormal #for deterministic: MVN = deltaMVN
    #sample top-level
    submean = phi[:G]
    chol = gamma_cov.cholesky()

    gamma = pyro.sample('gamma',dist.OMTMultivariateNormal(submean,chol),
                    infer={'is_auxiliary': True})


    #decompose gamma into specific values
    tmpgamma = gamma
    for pname in gamma_names:
        phat = gamma_parts[pname]
        elems = phat.nelement()
        #print(f"pname, phat: {pname}, {phat}")
        pdat, tmpgamma = tmpgamma[:elems], tmpgamma[elems:]
        pdat = transformations[pname](pdat)
        #print(f"adding {pname} from gamma ({elems}, {phat.size()})" )
        if pname not in fhat_data: #don't do frequentist params yet, see just below
            pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))

    for k,v in fhat_data.items():
        pyro.sample(k, dist.Delta(transformations[k](v)))


    #sample unit-level parameters, conditional on top-level ones
    global_indices = ts(range(G))
    base_gamma = gamma
    base_gamma_hat = phi[:G]
    ylist = []
    for i in range(N):
        #
        #

        precinct_indices = ts([G + i]) #norm_part[i], t_part[i]

        full_indices = torch.cat([global_indices, precinct_indices],0)

        #off-diagonals
        ll = hess.index_select(0,precinct_indices).index_select(1,global_indices)#lower left
        ur = hess.index_select(0,global_indices).index_select(1,precinct_indices)#upper right; could have just used transpose

        #diagonals
        lr = lowerblocks[i]#hess.index_select(0,precinct_indices).index_select(1,precinct_indices)#lower right
        ul = head_precision + torch.mm(torch.mm(ur,torch.inverse(lr)),ll) #upper left

        #print("dimensions",ll.size(),ur.size(),ul.size(),lr.size())
        full_precision = (
                torch.cat([
                    torch.cat([ul,ll],0),
                    torch.cat([ur,lr],0)
                ],1)
            )
        full_mean = phi.index_select(0,full_indices) #TODO: do in-place!
        new_mean, new_precision = conditional_normal(full_mean, full_precision, G, gamma)



        try:
            with poutine.scale(scale=1.):#weight):
                ylist.append( pyro.sample(f"y_{i}",
                                dist.OMTMultivariateNormal(new_mean,
                                        torch.inverse(new_precision).cholesky())
                                ,infer={'is_auxiliary': True}))
        except:
            print(new_precision)
            print(f"det:{np.linalg.det(new_precision.data.numpy())}")
            raise

    if N == full_N:
        t_units = full_units
    else:
        #print("full_ylist = ylist")
        t_units = chosen_units

    #print("guide t_part 2:",len(full_ylist))
    t_part = torch.cat([y.view(-1) for y in ylist],0)


    if False: #the debugging that led me to add evil_hack above
        fake_outcome = torch.sum(t_part**2)
        intermediate_vars = [dm,dt,ddf,full_tpart,
                            sub_tpart,hess,
                            submean,chol,gamma_cov] + ylist
        [eachvar.retain_grad() for eachvar in
            intermediate_vars + MLEvars
        ]
        fake_outcome.backward()
        print("Elementary:")
        print(torch.isnan(MLEvars[9].grad).nonzero())
        print(MLEvars[-2][240:250])
        for i in range(len(MLEvars)-1):
            print(i,":",MLEvars[i][240:245])
        import pdb; pdb.set_trace()
    #print("t_part",t_part.size())

    with t_units():
        pyro.sample("t_part", dist.Delta(t_part))
    #
    #print("end guide.",gamma[:3],mode_hat,nscale_hat,tscale_hat,hess[:5,:5],hess[-3:,-3:])
    #
    #print(".....1....",true_g_hat,gamma[-6:])
    #print(".....2....",gamma[-9:-6])


    if amortize:
        intermediate_vars = [dm,dt,ddf,full_tpart,
                            sub_tpart,hess,
                            submean,chol,gamma_cov] + ylist
        [eachvar.retain_grad() for eachvar in
            intermediate_vars + MLEvars
        ]
        def fix_m_grad():
            #print("fix_m_grad")
            if torch.any(torch.isnan(mode_hat.grad)):
                complain( "mode_hat.grad")
            if torch.any(torch.isnan(dm.grad)):
                complain("dm.grad")
                ftp = full_tpart
                stp = sub_tpart
                tp = t_part
                iv = intermediate_vars + MLEvars
                a,b,c,d = full_errors, dt, full_x-dm, ddf
                print("intermediate_vars")
                for eachvar in intermediate_vars: print(eachvar.grad)
                print("MLE vars")
                for eachvar in MLEvars: print(eachvar.grad)
                print(indices)
                #getMLE(full_errors[243].unsqueeze(-1),dt,full_x[243].unsqueeze(-1)-dm,ddf,return_intermediates=True)
                #for i in range(2,len(MLEvars)): print("var",i,"val",MLEvars[i][243],"grad",MLEvars[i].grad[243])
                #for i in range(2,len(MLEvars)): print(torch.isnan(MLEvars[i].grad).nonzero())
                import pdb; pdb.set_trace()
            else:
                yay("fix_m_grad")
                mode_hat.grad = mode_hat.grad + dm.grad
            mode_hat.grad[mode_hat.grad == float("Inf")] = 1e10
            mode_hat.grad[mode_hat.grad == float("-Inf")] = -1e10
            #print("mode_hat.grad",mode_hat.grad)
        mode_hat.fix_grad = fix_m_grad
        #
        def fix_t_grad():
            if torch.any(torch.isnan(ltscale_hat.grad)):
                complain( "ltscale_hat.grad")
                ftp = full_tpart
                stp = sub_tpart
                tp = t_part
                iv = intermediate_vars + MLEvars
                a,b,c,d = full_errors, dt, full_x-dm, ddf
                import pdb; pdb.set_trace()
            if torch.any(torch.isnan(dtr.grad)):
                complain( "dtr.grad")
                ftp = full_tpart
                stp = sub_tpart
                tp = t_part
                iv = intermediate_vars + MLEvars
                a,b,c,d = full_errors, dt, full_x-dm, ddf
                import pdb; pdb.set_trace()
            else:
                ltscale_hat.grad = ltscale_hat.grad + dtr.grad
            ltscale_hat.grad[ltscale_hat.grad == float("Inf")] = 1e10
            ltscale_hat.grad[ltscale_hat.grad == float("-Inf")] = -1e10
            #print("ltscale_hat.grad",ltscale_hat.grad)
        ltscale_hat.fix_grad = fix_t_grad

        def fix_df_grad():
            if torch.any(torch.isnan(ldfraw_hat.grad)):
                complain( "ldfraw_hat.grad")
                ftp = full_tpart
                stp = sub_tpart
                tp = t_part
                iv = intermediate_vars + MLEvars
                a,b,c,d = full_errors, dt, full_x-dm, ddf
                import pdb; pdb.set_trace()
            if torch.any(torch.isnan(ddfr.grad)):
                complain( "ddfr.grad")
                ftp = full_tpart
                stp = sub_tpart
                tp = t_part
                iv = intermediate_vars + MLEvars
                a,b,c,d = full_errors, dt, full_x-dm, ddf
                print("intermediate_vars")
                for eachvar in intermediate_vars: print(eachvar.grad)
                print("MLE vars")
                for eachvar in MLEvars: print(eachvar.grad)
                print(indices)
                #for i in range(len(MLEvars)): if sum(MLEvars[i].size()) > 1: print("var",i,"val",MLEvars[i][243],"grad",MLEvars[i].grad[243])
                import pdb; pdb.set_trace()
            else:
                ldfraw_hat.grad = ldfraw_hat.grad + ddfr.grad
            ldfraw_hat.grad[ldfraw_hat.grad == float("Inf")] = 1e10
            ldfraw_hat.grad[ldfraw_hat.grad == float("-Inf")] = -1e10
            #print("ldfraw_hat.grad",ldfraw_hat.grad)
        ldfraw_hat.fix_grad = fix_df_grad

    if save_data is not None:
        save_data.update(ahat_data=hat_data,
                        logPosterior=logPosterior,
                        raw_hessian=hess,
                        fixed_hessian_upper = head_precision,
                        head_adjustment=head_adjustment,
                        grad=grad,
                        df = ddf,
                        globalpsiraw=globalpsiraw,
                        latentpsiraw=latentpsiraw)
    return(save_data)
    #

def unamortized_laplace(*args,**kwargs):
    return(laplace_guide(*args,amortize=False,**kwargs))


#     #
# def amortized_meanfield(N,full_N,indices,x,full_x,errors,full_errors,maxError,save_data,weight=1.,scalehyper=ts(4.),tailhyper=ts(10.)):
#     #print("guide2 start")
#     #
#     hat_data = OrderedDict()
#     mode_hat = pyro.param("mode_hat",ts(0.))
#     hat_data.update(modal_effect=mode_hat)
#
#     nscale_hat = pyro.param("nscale_hat",ts(1.)) #log this? nah; just use "out of box"
#     hat_data.update(norm_scale=nscale_hat)
#
#     tscale_hat = pyro.param("tscale_hat",ts(1.))
#     hat_data.update(t_scale=tscale_hat)
#
#     normpart, tpart = getMLE(nscale_hat, tscale_hat, errors, x)
#
#     true_n_hat = normpart * nscale_hat / (nscale_hat + errors)
#     hat_data.update(norm_part=true_n_hat)
#
#     true_g_hat = tpart
#     hat_data.update(t_part=true_g_hat)
#
#     #Get hessian
#     phi = torch.cat([gammaPart.view(-1) for gammaPart in hat_data.values()],0)
#     nparams = len(phi)
#
#     gamma_scale = pyro.param("gamma_scale",torch.ones(nparams))
#
#     #declare global-level psi params
#     gamma = pyro.sample('gamma',
#                     dist.Normal(phi,torch.abs(gamma_scale)).independent(1),
#                     infer={'is_auxiliary': True})
#
#     #decompose gamma into specific values
#     tmpgamma = gamma
#     for pname, phat in hat_data.items():
#         elems = phat.nelement()
#         pdat, tmpgamma = tmpgamma[:elems], tmpgamma[elems:]
#         #print(f"adding {pname} from gamma ({elems}, {phat.size()})" )
#
#         pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))
#     #

def meanfield(N,full_N,indices,x,full_x,errors,full_errors,maxError,
                        save_data=None,
                        weight=1.,scalehyper=ts(4.),tailhyper=ts(10.)):

    mode_hat = pyro.param("mode_hat",ts(0.))
    mode_sigma = pyro.param("mode_sigma",torch.ones([]),constraint=constraints.positive)
    mode = pyro.sample("modal_effect",dist.Normal(mode_hat,mode_sigma))

    ltscale_hat = pyro.param("ltscale_hat",ts(0.))
    ltscale_sigma = pyro.param("ltscale_sigma",torch.ones([]),constraint=constraints.positive)
    lts = pyro.sample("log_t_scale",dist.Normal(ltscale_hat,ltscale_sigma),
            infer={'is_auxiliary': True})
    t_scale_raw = pyro.sample("t_scale_raw",dist.Delta(torch.exp(lts)))

    #
    ldfraw_hat = pyro.param("ldfraw_hat",ts(0.))
    ldfraw_sigma = pyro.param("ldfraw_sigma",torch.ones([]),constraint=constraints.positive)
    ldf = pyro.sample("log_df",dist.Normal(ldfraw_hat,ldfraw_sigma),
            infer={'is_auxiliary': True})
    t_scale_raw = pyro.sample("dfraw",dist.Delta(torch.exp(ldf)))
    #hat_data.update(fhat_data) #include frequentist value in Hessian
        #can't, no Hessians of gamma functions

    t_part_hat = pyro.param("t_part_hat",torch.zeros(full_N))
    t_part_sigma = pyro.param("t_part_sigma",torch.ones(full_N),constraint=constraints.positive)
    tph = t_part_hat.index_select(0,indices)
    tps = t_part_sigma.index_select(0,indices)
    with poutine.scale(scale=weight):
        with pyro.plate('units',N):
            t_part = pyro.sample("t_part",dist.Normal(tph,tps))



def compile_params(source):
    raw = source[0]
    compiled = dict(modal_effect = ts(raw["modal_effect"]),
                    df=ts(math.log(raw["df"] - MIN_DF)),
                    t_scale=ts(math.log(raw["t_scale"] - MIN_SIGMA_OVER_S)))
    return compiled
#
#fake_x = model(N,full_N,indices,echs_x,errors,fixedParams = tdom_params)
#print("Fake:",fake_x)

#autoguide = AutoDiagonalNormal(model)
guides = OrderedDict(
                    meanfield=meanfield,
                    amortized_laplace = laplace_guide,
                    unamortized_laplace = unamortized_laplace,
                    )

class FakeSink(object):
    def write(self, *args):
        pass
    def writelines(self, *args):
        pass
    def close(self, *args):
        pass

def getLaplaceParams():
    store = pyro.get_param_store()
    result = []
    for item in ("mode_hat","ltscale_hat","ldfraw_hat","globalpsi","latentpsi","EMPTY_PLACEHOLDER"):
        try:
            result.append(store[item])
        except:
            result.append("")

    return result

def getMeanfieldParams():
    store = pyro.get_param_store()
    result = []
    for item in ("mode_hat","ltscale_hat","ldfraw_hat","mode_sigma","ltscale_sigma","ldfraw_sigma"):
        try:
            result.append(store[item])
        except:
            result.append("")

    return result

def floaty(l):
    return ts([float(i) for i in l])

def nameWithParams(filebase, sourceparams, errors, S = None):
    trueparams = sourceparams[0]
    if S is None:
        filename = (f"{filebase}_N{len(errors)}_mu{trueparams['modal_effect']}"+


            f"_sigma{trueparams['t_scale']}_nu{trueparams['df']}.csv")
    else:
        filename = (f"{filebase}_N{len(errors)}_S{S}_mu{trueparams['modal_effect']}"+
            f"_sigma{trueparams['t_scale']}_nu{trueparams['df']}.csv")
    return filename

def createECHSScenario(sourceparams,
            errors=echs_errors,
            junkData=echs_x,
            filebase="testresults/scenario"):
    filename = nameWithParams(filebase, sourceparams, errors)
    try:
        print("1")
        with open(filename,"r") as file:
            reader = csv.reader(file)
            header = next(reader)
            lines = list(reader)
            s,t,x = zip(*lines)
        s,x = (floaty(s), floaty(x))
        print(filename, "from file")
    except Exception as e:
        print("exception:",e)
        maxError = torch.max(errors)
        save_data = dict()
        t,x = model(N,N,  None,   junkData,junkData,
                    errors,errors,
                    maxError,save_data,1.,
                    fixedParams = compile_params(sourceparams))
                    #N,full_N,indices,x,     full_x,errors,full_errors,maxError,weight=1.,scalehyper=ts(4.),tailhyper=ts(10.)):

        with open(filename,"w") as file:
            writer = csv.writer(file)
            writer.writerow([u"s",u"t",u"x"])
            for an_s,a_t,an_x in zip(errors,t,x):
                writer.writerow([float(an_s),float(a_t),float(an_x)])
        print(filename, "created")
    print(len(x))


    print(x[:4])
    return x


def createScenario(sourceparams,
            nSites = N_SAMPLES,
            errorDistribution = torch.distributions.Gamma(4,8),
            filebase="testresults/scenario"):
    errors = errorDistribution.sample(torch.Size([nSites]))
    errors[(errors>1).nonzero()] = 1.
    junkData = errorDistribution.sample(torch.Size([nSites])) #gotta have something, even though it's trashed later
    filename = nameWithParams(filebase, sourceparams, errors)
    N = N_SAMPLES
    try:
        print("1")
        with open(filename,"r") as file:
            reader = csv.reader(file)
            header = next(reader)
            lines = list(reader)
            s,t,x = zip(*lines)
        s,t,x = (floaty(s), floaty(t), floaty(x))
        print(filename, "from file")
    except Exception as e:
        print("exception:",e)
        maxError = torch.max(errors)
        save_data = dict()
        t,x = model(N,N,  None,   junkData,junkData,
                    errors,errors,
                    maxError,save_data,1.,
                    fixedParams = compile_params(sourceparams))
                    #N,full_N,indices,x,     full_x,errors,full_errors,maxError,weight=1.,scalehyper=ts(4.),tailhyper=ts(10.)):
        assert not (os.path.exists(filename)) #don't just blindly overwrite!
        with open(filename,"w", newline="\n") as file:
            writer = csv.writer(file)
            writer.writerow([u"s",u"t",u"x"])
            for an_s,a_t,an_x in zip(errors,t,x):
                writer.writerow([float(an_s),float(a_t),float(an_x)])
        print(filename, "created")
    print(len(x))
    print(x[:4])
    return (x,errors)


def createECHSScenario2(sourceparams,
            errors=echs_errors,
            junkData=echs_x,
            filebase="testresults/scenario"):
    filename = nameWithParams(filebase, sourceparams, errors)
    try:
        print("1")
        with open(filename,"r") as file:
            reader = csv.reader(file)
            header = next(reader)
            lines = list(reader)
            s,t,x = zip(*lines)
        s,x = (floaty(s), floaty(x))
        print(filename, "from file")
    except Exception as e:
        print("exception:",e)
        maxError = torch.max(errors)
        save_data = dict()
        t,x = model(N,N,  None,   junkData,junkData,
                    errors,errors,
                    maxError,save_data,1.,
                    fixedParams = compile_params(sourceparams))
                    #N,full_N,indices,x,     full_x,errors,full_errors,maxError,weight=1.,scalehyper=ts(4.),tailhyper=ts(10.)):

        with open(filename,"w") as file:
            writer = csv.writer(file)
            writer.writerow([u"s",u"t",u"x"])
            for an_s,a_t,an_x in zip(errors,t,x):
                writer.writerow([float(an_s),float(a_t),float(an_x)])
        print(filename, "created")
    print(len(x))
    print(x[:4])
    return x


def jsonizable(thing):
    #print("jsonizable...",thing)
    #print("type",type(thing), type(thing) is FUCKING_TENSOR_TYPE, type(thing) == type(torch.tensor(1.)))
    if isinstance(thing, Mapping):
        return dict([(key, jsonizable(val)) for (key, val) in thing.items()])
    elif type(thing) is FUCKING_TENSOR_TYPE:
        t = thing.tolist()
        #print("not tense",t)
        return(t)
    return thing

def jsonize(thing):
    #print("jsonizing")
    t = jsonizable(thing)

    #print("jsonizing 2", t)
    return(json.dumps(t, indent=2, sort_keys=True))

    #print("jsonized")

def saveFit(guidename, save_data,
        sourceparams, errors, S, nparticles,
        filebase="testresults/fit_"):
    i = 0
    while True:
        filename = nameWithParams(filebase+guidename+"_"+str(i)+"_parts"+str(nparticles),
                sourceparams, errors, S)
        if not os.path.exists(filename):
            break
        print("file exists:",filename)
        i += 1
    with open(filename, "w") as output:
        output.write(jsonize(save_data))
    #print("saveFit", filename)

def addNumericalHessianRow(ctr, sides, where): #`where` is number of columns before new elem
    old_hess = ctr["raw_hessian"]
    dims = len(old_hess) + 1
    new_hess = torch.zeros(dims, dims)
    new_hess[:where,:where] = old_hess[:where,:where]
    new_hess[where+1:,where+1:] = old_hess[where:,where:]
    new_hess[where+1:,:where] = old_hess[where:,:where]
    new_hess[:where,where+1:] = old_hess[:where,where:]

    diff = torch.tensor(0.)
    for (delta, side) in sides.items():
        diff -= (side["logPosterior"] - ctr["logPosterior"]) / delta**2 #kinda magic but it works
    if diff<0:
        diff = .1 #Utterly arbitrary. TODO: something principled????
    new_hess[where,where] = diff
    #TODO: grads

    new_result = dict(ctr)
    new_result.update(raw_hessian = new_hess)
    return new_result


tctr = {"raw_hessian":torch.ones(2,2), "logPosterior":0}
tside = {"logPosterior":-.125}
tsides = {-.5:tside,.5:tside}
tnewhess = addNumericalHessianRow(tctr,tsides,1)["raw_hessian"]
for i in range(3):
    for j in range(3):
        assert tnewhess[i,j] == (1+i+j) % 2
#print("addNumericalHessianRow tested")

def writeableList(item):
    if type(item) is str:
        return [item]
    if type(item) is FUCKING_TENSOR_TYPE and sum(item.size()) > 0:
        return [float(subitem) for subitem in item]
    return [float(item)]

def myWriteRow(writer,row):
    writer.writerow(list(itertools.chain(*[writeableList(item) for item in row])))

def trainGuide(guidename = "laplace",
            nparticles = 1,
            sourceparams = tdom_fat_params,
            filename = None,
            errors=errors,
            subsample_N=SUBSAMPLE_N,
            N = N_SAMPLES):

    guide = guides[guidename]
    if sourceparams == None:
        (x,errors) = (echs_x, echs_errors)
        N = len(x)
        subsample_N = min(N,subsample_N)
        sourceparams = dummy_echs_params
    else:
        (x,errors) = createScenario(sourceparams, N)

    weight = N * 1. / subsample_N
    print("Sizes:",x.size(),errors.size())

    maxError = torch.max(errors)

    print("guidename",guidename)
    print("sourceparams",sourceparams)
    print("sizes",x.size(),N,subsample_N,weight)

    alreadyExists = True
    if filename is None:
        file = FakeSink()

    else:
        alreadyExists = os.path.exists(filename)
        file = open(filename,"a")
    writer = csv.writer(file)
    if not(alreadyExists):
        base_line_names = ["guidename", "runtime","truemu",
            "truenu","truesigma","i", "time", "loss", "mustar", "sigmastar", "nustar", "XX", "YY", "ZZ"]
        writer.writerow(base_line_names)

    #guide = guide2
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO(nparticles))
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'betas': (0.99,0.9999)}), Trace_ELBO(nparticles)) #.72
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'clip_norm': 5.0}), Trace_ELBO(nparticles)) #.66
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'weight_decay': ...}), Trace_ELBO(nparticles))
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'eps': 1e-5}), Trace_ELBO(nparticles))
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'eps': 1e-10}), Trace_ELBO(nparticles))
    #svi = SVI(model, guide, AdagradRMSProp({}), Trace_ELBO(nparticles))


    for subsample_n in [subsample_N]: #,N]:
        svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'betas': (0.8,0.9)}), Trace_ELBO(nparticles)) #?
        pyro.clear_param_store()
        losses = []
        mean_losses = [] #(moving average)
        runtime = time.time()
        base_line = [guidename, runtime] + [
                        sourceparams[0][item] for item in ("modal_effect",
                                        "df","t_scale")]
        for i in range(MAX_OPTIM_STEPS):
            indices = torch.randperm(N)[:subsample_n]
            save_data = dict()
            # print("stepping",subsample_n,N,indices.size(),
            #                 x.index_select(0,indices).size(),x.size(),
            #                 errors.index_select(0,indices).size(),errors.size(),
            #                 maxError,
            #                 save_data,weight,ts(10.),ts(10.))
            loss = svi.step(subsample_n,N,indices,
                            x.index_select(0,indices),x,
                            errors.index_select(0,indices),errors,
                            maxError,
                            save_data,weight,ts(10.),ts(10.))
                        #N,full_N,indices,
                        #x,     full_x,
                        #errors,full_errors,
                        #maxError,weight=1.,scalehyper=ts(4.),tailhyper=ts(10.)):

            if len(losses)==0:
                mean_losses.append(loss)
            else:
                mean_losses.append((mean_losses[-1] * 49. + loss) / 50.)
            losses.append(loss)
            if i % 10 == 0:
                try:
                    myWriteRow(writer,base_line + [i, time.time(), loss] + getLaplaceParams())
                except:
                    myWriteRow(writer,base_line + [i, time.time(), loss] + getMeanfieldParams())
                reload(go_or_nogo)
                go_or_nogo.demoprintstuff(i,loss)
                try:
                    if mean_losses[-1] > mean_losses[-500]:
                        break
                except:
                    pass
                if go_or_nogo.go:
                    pass
                else:
                    break

        ##

        print("Final mean_losses:",mean_losses[-1])
        plt.plot(losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')

        #print("save_data",save_data)
        if not save_data:
            #meanfield
            save_data = dict(pyro.get_param_store())
        else:
            all_indices = torch.tensor(range(N))
            #once more, without subsampling
            tmp_data = dict()
            fitted_model_info = guide(N,N,all_indices,
                            x,x,
                            errors,errors,
                            maxError,
                            tmp_data,1.,ts(10.),ts(10.))
            perturbed_datas = dict() #"data" is singular, deal with it
            for adj in [DF_ADJ_QUANTUM, -DF_ADJ_QUANTUM]:

                tmp_data = dict(df_adj=adj)
                perturbed_datas[adj] = guide(N,N,all_indices,
                                x,x,
                                errors,errors,
                                maxError,
                                tmp_data,1.,ts(10.),ts(10.))
            save_data = addNumericalHessianRow(fitted_model_info, perturbed_datas, 2)

        global COMPLAINTS_REMAINING
        global YAYS_REMAINING
        save_data.update(ycomplaints = (BASE_COMPLAINTS_REMAINING - COMPLAINTS_REMAINING) / 3,
                        yays = (BASE_YAYS_REMAINING - YAYS_REMAINING) / 3,)

        COMPLAINTS_REMAINING = BASE_COMPLAINTS_REMAINING
        YAYS_REMAINING = BASE_YAYS_REMAINING

        saveFit(guidename, save_data, sourceparams, errors, subsample_n, nparticles)
        ##

        print("guidename",guidename)
        print("sourceparams",sourceparams)
        for (key, val) in sorted(pyro.get_param_store().items()):
            if sum(val.size()) > 1:
                print(f"{key}:\n{val[:10]} (10 elems)")
            else:
                print(f"{key}:\n{val}")
