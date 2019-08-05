#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

from importlib import reload
import csv
import time
import math
import random #Mixing seeds — not reproducible — TODO:fix
import itertools

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
ts = torch.tensor

pyro.enable_validation(True)
pyro.set_rng_seed(0)



SAFETY_MULT = 1.01
LOG_BASE_PSI = math.log(BASE_PSI)

EULER_CONSTANT = 0.5772156649015328606065120900824024310421
GUMBEL_SD = math.pi/math.sqrt(6.)


COMPLAINTS_REMAINING = 10

def complain(*args):
    global COMPLAINTS_REMAINING
    COMPLAINTS_REMAINING -= 1
    if COMPLAINTS_REMAINING >= 0:
        print("complaint",COMPLAINTS_REMAINING,*args)
    if COMPLAINTS_REMAINING % 20 == 0:
        print("complaint",COMPLAINTS_REMAINING)

def model(N,full_N,indices,effects,full_effects,errors,full_errors,maxError,weight=1.,
            scalehyper=ts(4.),tailhyper=ts(10.),
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
        full_units = chosen_units
    else:
        full_units = pyro.plate('full_units',full_N)


    if fixedParams is not None:
        pass#print("model for fake")
    else:
        pass#print("model for real")

    #prior on μ_x
    modal_effect = pyro.sample('modal_effect',dist.Normal(ts(0.),ts(20.)))

    #prior on sd(τ)
    t_scale = maxError/2+pyro.sample('t_scale_raw',dist.Exponential(torch.ones(1)/torch.abs(scalehyper)))

    #prior on df
    df = 2. + pyro.sample('dfraw',dist.Exponential(torch.ones(1)/torch.abs(scalehyper)))

    if fixedParams is not None:
        print("generating data",fixedParams)
        #
        modal_effect = fixedParams['modal_effect']
        t_scale = maxError/2+torch.exp(fixedParams['t_scale'])
        df = (2. + torch.exp(fixedParams['df'])).requires_grad_()

    if N==full_N:
        with pyro.plate('2full_units',full_N): #I hate you! but this magic works?
            t_part = pyro.sample('t_part',dist.StudentT(df,torch.zeros(full_N),ts(1.)))
    else:
        with full_units:
            t_part = pyro.sample('t_part',dist.StudentT(df,torch.zeros(full_N),ts(1.)))

    #Latent true values (offset)
    if indices is not None:
        #print("indices",indices,full_N,t_part.size())
        good_parts = t_part.index_select(0,indices)
    else:
        good_parts = t_part
    truth = modal_effect + good_parts * t_scale

    #Observations conditional on truth (likelihood)
    if fixedParams is not None:
        print("creating",modal_effect.size(),t_part.size(),t_scale.size(),truth.size(),errors.size())
        observations = pyro.sample('observations', dist.Normal(truth,errors).to_event(1))
        print("Not none - creating data.")
    else:
        try:
            with chosen_units():
                observations = pyro.sample('observations', dist.Normal(truth,errors), obs=effects)
        except:
            print("ERROR", truth, errors)
            print("ERROR", modal_effect, norm_scale, t_scale)
            raise

    if fixedParams is not None:
        return observations.detach()
    #print("end model",modal_effect,norm_scale,t_scale,t_part)


def laplace(N,full_N,indices,effects,full_effects,errors,full_errors,maxError,
            weight=1.,scalehyper=ts(4.),tailhyper=ts(10.),
            deterministic=False):


    units_plate = pyro.plate('units',N)
    @contextlib.contextmanager
    def chosen_units():
        with units_plate as n, poutine.scale(scale=weight) as nscale:
            yield n


    hat_data = OrderedDict() #
    transformations = defaultdict(lambda: lambda x: x) #factory of identity functions
    mode_hat = pyro.param("mode_hat",ts(0.))
    hat_data.update(modal_effect=mode_hat)

    ltscale_hat = pyro.param("ltscale_hat",ts(1.))
    hat_data.update(t_scale_raw=ltscale_hat)
    transformations.update(t_scale_raw=torch.exp)

    ldfraw_hat = pyro.param("ldfraw_hat",ts(1.))

    true_t_hat = pyro.param("true_t_hat",(full_effects/2).detach())
    if N == full_N:
        sub_t_hat = true_t_hat
    else:
        sub_t_hat = true_t_hat.index_select(0,indices)
        print(indices,N,full_N)
        static_mask = torch.ones(full_N) - torch.sparse.FloatTensor(indices.view(1,-1), torch.ones(N), torch.Size([full_N]))
        def reconstitute(live_parts):
            return (static_mask * true_t_hat +
                    torch.sparse.FloatTensor(indices.view(1,-1), live_parts, torch.Size([full_N])))
        transformations.update(t_part=reconstitute)
    hat_data.update(t_part=sub_t_hat)

    #Get hessian
    thetaMean = torch.cat([transformations[pname](thetaPart).view(-1) for pname, thetaPart in hat_data.items()],0)
    transformed_hat = dict((pname,transformations[pname](thetaPart)) for pname, thetaPart in hat_data.items())


    transformed_hat.update(dfraw=torch.exp(ldfraw_hat))

    #print("transformed_hat",transformed_hat)

    hessCenter = pyro.condition(model,transformed_hat)
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)(N,full_N,indices,effects,full_effects,errors,maxError,weight,scalehyper,tailhyper) #*args,**kwargs)
    logPosterior = blockedTrace.log_prob_sum()
    #print("lap",logPosterior)
    # for k,value in reversed(list(transformed_hat.items())):#.reverse():
    #     for i in range(10):
    #         print()
    #     print(k)
    #     print(myhessian.hessian(logPosterior,[value]))
    Info = -myhessian.hessian(logPosterior, hat_data.values())#, allow_unused=True)
    #print("det:",np.linalg.det(Info.data.numpy()))
    #print(Info)
    posDef = rescaledSDD(Info)


    if False: #print determinant
        det = np.linalg.det(posDef.data.numpy())

        #print("Got hessian")
        if math.isinf(det) or (det<=0):
            print("adet:",det)
            print("aInf:",posDef)
            corr = infoToM(Info,debug=True)
            print(torch.diag(corr))
            print(corr)#torch.sum(torch.diag(corr)-torch.sum(corr)))
            print("adet3:",np.linalg.det(posDef[:3,:3].data.numpy()))
            print("adet3:",np.linalg.det(posDef[:5,:5].data.numpy()))

    #declare global-level psi params

    if deterministic:
        MVN = deltaMVN
    else:
        MVN = dist.MultivariateNormal

    try:
        theta = pyro.sample('theta',
                        MVN(thetaMean,
                                    precision_matrix=posDef),
                        infer={'is_auxiliary': True})
    except:
        try:
            theta = pyro.sample('theta',
                            MVN(thetaMean,
                                        precision_matrix=rescaledSDD(Info,strong=True)),
                            infer={'is_auxiliary': True})
        except:
            infoToM(Info,debug=True)
            raise


    #decompose theta into specific values
    tmptheta = theta
    for pname, phat in hat_data.items():
        elems = phat.nelement()
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        pdat = transformations[pname](pdat)
        #print(f"adding {pname} from theta ({elems}, {phat.size()})" )

        if pname=="t_part":
            if N==full_N:
                reconstituted_t_hat = pdat
            else:
                reconstituted_t_hat = (static_mask * true_t_hat +
                        torch.sparse.FloatTensor(indices.view(1,-1), pdat, torch.Size([full_N])))
            with chosen_units():
                pyro.sample(pname, dist.Delta(reconstituted_t_hat))
        else:
            pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))

    pyro.sample("dfraw", dist.Delta(transformed_hat["dfraw"]))
    #

def deterministicLaplace(*args, **kwargs):
    return laplace(*args, deterministic=True, **kwargs)

def cuberoot(x):
    return torch.sign(x) * torch.abs(x) ** (1./3.)

def getMLE(nscale, tscale, obs, df):
    #assert getDiscriminant(nscale, tscale, obs, df) < 0 #only one root
    try:
        assert torch.all(tscale / nscale > math.sqrt((df + 1)/df/8)) #only one root, regardless of obs; redundant, stronger
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

    x = cuberoot(q + insqrt) + cuberoot(q - insqrt) + p

    try:
        assert approx_eq(x**3 + b * x**2 + c * x + d,torch.zeros(1))
    except:
        complain(
                "assert approx_eq(",x**3, b * x**2, c * x, d)
        complain(
                "assert2 approx_eq(",x**3 + b * x**2 + c * x + d)
        # raise
    # print("getMLE(",nscale, tscale, obs, df)
    # print("getMLE2(",b, c, d)
    # print("getMLE3(",p, q, r, inner, insqrt, x)
    # print("getMLE3(", (q - insqrt), (q - insqrt) ** (1./3.))
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



LAMBERT_MAX_ITERS = 10
LAMBERT_TOL = 1e-2

def get_unconditional_cov(full_precision, n):
    #TODO: more efficient
    return(torch.inverse(full_precision)[:n,:n])

def amortized_laplace(N,full_N,indices,effects,full_effects,errors,full_errors,maxError,
                        weight=1.,scalehyper=ts(4.),tailhyper=ts(10.),
                        deterministic=False):
    #print("amortized_laplace:",N,len(effects),len(errors),weight)


    units_plate = pyro.plate('units',N)
    @contextlib.contextmanager
    def chosen_units():
        with units_plate as n, poutine.scale(scale=weight) as nscale:
            yield n

    if N==full_N:
        full_units = chosen_units
    else:
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

    ldfraw_hat = pyro.param("ldfraw_hat",ts(0.))
    fhat_data.update(dfraw=ldfraw_hat)
    transformations.update(dfraw=torch.exp)


    dm = mode_hat.detach().requires_grad_()
    dtr = ltscale_hat.detach().requires_grad_() #detached... raw
    ddfr = ldfraw_hat.detach().requires_grad_() #detached... raw

    dt =  maxError/2 + torch.exp(dtr) #detached, cook
    ddf = 2. + torch.exp(ddfr) #detached, cook


    #print("amortized_laplace:",torch.max(errors),maxError,torch.max(errors)/dt)
    full_tpart = getMLE(full_errors, dt, full_effects-dm, ddf) / dt
    if N == full_N:
        sub_tpart = full_tpart
    else:
        sub_tpart = full_tpart.index_select(0,indices)
        static_mask = torch.ones(full_N) - torch.sparse.FloatTensor(indices.view(1,-1), torch.ones(N), torch.Size([full_N]))

    #print("tpart:",tpart)

    #true_g_hat.requires_grad = True
    theta_names = list(hat_data.keys())
    hat_data.update(t_part=sub_tpart)

    #Get hessian
    thetaMean = torch.cat([thetaPart.view(-1) for thetaPart in hat_data.values()],0)

    conditioner = dict()
    conditioner.update((k,transformations[k](v)) for k, v in itertools.chain(hat_data.items(), fhat_data.items()))
    hessCenter = pyro.condition(model,conditioner)
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)(N,N,None,effects,effects,errors,errors,maxError,weight,scalehyper,tailhyper) #*args,**kwargs)
    logPosterior = blockedTrace.log_prob_sum()
    theta_parts = dict((theta_name,hat_data[theta_name]) for theta_name in theta_names)
    Info = -myhessian.hessian(logPosterior, hat_data.values())#, allow_unused=True)


    lpsi = pyro.param("lpsi",torch.zeros(len(Info)) + LOG_BASE_PSI)
    #ensure positive definite
    big_hessian = Info + infoToM(Info, torch.exp(lpsi))


    if False: #code for testing hessian grads
        dumb_loss = torch.sum(big_hessian)
        print(f"mode_hat.grad:{mode_hat.grad}; dm.grad:{ dm.grad}")
        dumb_loss.backward(retain_graph=True)
        print(f"mode_hat.grad:{mode_hat.grad}; dm.grad:{ dm.grad}")


    if False: #print determinant
        det = np.linalg.det(big_hessian.data.numpy())
        print("det:",det)

        #print("Got hessian")
        if math.isinf(det):
            print("Inf:",big_hessian)
            print("det3:",np.linalg.det(Info[:3,:3].data.numpy()))
            print("det3:",np.linalg.det(Info[:5,:5].data.numpy()))

    #count parameters
    usedup = int(sum(theta_parts[pname].nelement() for pname in theta_names))

    #invert matrix (maybe later, smart)
    theta_cov = get_unconditional_cov(big_hessian,usedup)

    if deterministic:
        MVN = deltaMVN
    else:
        MVN = dist.MultivariateNormal
    #sample top-level parameters
    theta = pyro.sample('theta',
                    MVN(thetaMean[:usedup],
                                theta_cov),
                    infer={'is_auxiliary': True})

    #decompose theta into specific values
    tmptheta = theta
    for pname in theta_names:
        phat = theta_parts[pname]
        elems = phat.nelement()
        #print(f"pname, phat: {pname}, {phat}")
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        pdat = transformations[pname](pdat)
        #print(f"adding {pname} from theta ({elems}, {phat.size()})" )

        pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))

    for k,v in fhat_data.items():
        pyro.sample(k, dist.Delta(transformations[k](v)))


    #sample unit-level parameters, conditional on top-level ones
    global_indices = ts(range(usedup))
    base_theta = theta
    base_theta_hat = thetaMean[:usedup]
    ylist = []
    for i in range(N):
        #
        #

        precinct_indices = ts([usedup + i]) #norm_part[i], t_part[i]

        full_indices = torch.cat([global_indices, precinct_indices],0)

        full_precision = big_hessian.index_select(0,full_indices).index_select(1,full_indices) #TODO: do in-place?
        full_mean = thetaMean.index_select(0,full_indices) #TODO: do in-place!
        new_mean, new_precision = conditional_normal(full_mean, full_precision, usedup, theta)



        try:
            with poutine.scale(scale=weight):
                ylist.append( pyro.sample(f"y_{i}",
                                MVN(new_mean, precision_matrix=new_precision),
                                infer={'is_auxiliary': True}))
        except:
            print(new_precision)
            print(f"det:{np.linalg.det(new_precision.data.numpy())}")
            raise

    if N == full_N:
        t_part = torch.cat([y.view(-1) for y in ylist],0)
    else:
        full_ylist = [None] * full_N
        for i in range(full_N):
            if static_mask[i] > 0:
                full_ylist[i] = full_tpart[i]
        for i in range(N):
            full_ylist[indices[i]] = ylist[i]

        t_part = torch.cat([y.view(-1) for y in full_ylist],0)

    #print("t_part",t_part.size())
    with full_units():
        pyro.sample("t_part", dist.Delta(t_part))
    #
    #print("end guide.",theta[:3],mode_hat,nscale_hat,tscale_hat,Info[:5,:5],Info[-3:,-3:])
    #
    #print(".....1....",true_g_hat,theta[-6:])
    #print(".....2....",theta[-9:-6])
    def fix_m_grad():
        if torch.any(torch.isnan(mode_hat.grad)):
            complain( "mode_hat.grad")
        if torch.any(torch.isnan(dm.grad)):
            complain("dm.grad")
        else:
            mode_hat.grad = mode_hat.grad + dm.grad
        mode_hat.grad[mode_hat.grad == float("Inf")] = 1e10
        mode_hat.grad[mode_hat.grad == float("-Inf")] = -1e10
        #print("mode_hat.grad",mode_hat.grad)
    mode_hat.fix_grad = fix_m_grad
    #
    def fix_t_grad():
        if torch.any(torch.isnan(ltscale_hat.grad)):
            complain( "ltscale_hat.grad")
        if torch.any(torch.isnan(dtr.grad)):
            complain( "dtr.grad")
        else:
            ltscale_hat.grad = ltscale_hat.grad + dtr.grad
        ltscale_hat.grad[ltscale_hat.grad == float("Inf")] = 1e10
        ltscale_hat.grad[ltscale_hat.grad == float("-Inf")] = -1e10
        #print("ltscale_hat.grad",ltscale_hat.grad)
    ltscale_hat.fix_grad = fix_t_grad

    def fix_df_grad():
        if torch.any(torch.isnan(ldfraw_hat.grad)):
            complain( "ldfraw_hat.grad")
        if torch.any(torch.isnan(ddfr.grad)):
            complain( "ddfr.grad")
        else:
            ldfraw_hat.grad = ldfraw_hat.grad + ddfr.grad
        ldfraw_hat.grad[ldfraw_hat.grad == float("Inf")] = 1e10
        ldfraw_hat.grad[ldfraw_hat.grad == float("-Inf")] = -1e10
        #print("ldfraw_hat.grad",ldfraw_hat.grad)
    ldfraw_hat.fix_grad = fix_df_grad
    #

def amortized_deterministic_laplace(N,full_N,indices,effects,full_effects,errors,full_errors,maxError,weight=1.,scalehyper=ts(4.),tailhyper=ts(10.)):
    #print("amortized_laplace:",N,len(effects),len(errors),weight)


    units_plate = pyro.plate('units',N)
    @contextlib.contextmanager
    def chosen_units():
        with units_plate as n, poutine.scale(scale=weight) as nscale:
            yield n

    if N==full_N:
        full_units = chosen_units
    else:
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

    ldfraw_hat = pyro.param("ldfraw_hat",ts(0.))
    fhat_data.update(dfraw=ldfraw_hat)
    transformations.update(dfraw=torch.exp)


    dm = mode_hat.detach().requires_grad_()
    dtr = ltscale_hat.detach().requires_grad_() #detached... raw
    ddfr = ldfraw_hat.detach().requires_grad_() #detached... raw

    dt =  maxError/2 + torch.exp(dtr) #detached, cook
    ddf = 2. + torch.exp(ddfr) #detached, cook


    #print("amortized_laplace:",torch.max(errors),maxError,torch.max(errors)/dt)
    full_tpart = getMLE(full_errors, dt, full_effects-dm, ddf) / dt
    tpart = full_tpart.index_select(0,indices)
    #print("tpart:",tpart)

    true_t_hat = tpart#.detach()
    #true_g_hat.requires_grad = True
    theta_names = list(hat_data.keys())
    hat_data.update(t_part=true_t_hat)
    param_names = list(hat_data.keys())

    #Get hessian
    thetaMean = torch.cat([thetaPart.view(-1) for thetaPart in hat_data.values()],0)

    conditioner = dict()
    conditioner.update((k,transformations[k](v)) for k, v in itertools.chain(hat_data.items(), fhat_data.items()))
    hessCenter = pyro.condition(model,conditioner)
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)(N,N,None,effects,effects,errors,errors,maxError,weight,scalehyper,tailhyper) #*args,**kwargs)
    logPosterior = blockedTrace.log_prob_sum()
    theta_parts = dict((theta_name,hat_data[theta_name]) for theta_name in theta_names)
    Info = -myhessian.hessian(logPosterior, hat_data.values())#, allow_unused=True)


    lpsi = pyro.param("lpsi",torch.zeros(len(Info)) + LOG_BASE_PSI)
    #ensure positive definite
    big_hessian = Info + infoToM(Info, torch.exp(lpsi))


    if False: #code for testing hessian grads
        dumb_loss = torch.sum(big_hessian)
        print(f"mode_hat.grad:{mode_hat.grad}; dm.grad:{ dm.grad}")
        dumb_loss.backward(retain_graph=True)
        print(f"mode_hat.grad:{mode_hat.grad}; dm.grad:{ dm.grad}")


    if False: #print determinant
        det = np.linalg.det(big_hessian.data.numpy())
        print("det:",det)

        #print("Got hessian")
        if math.isinf(det):
            print("Inf:",big_hessian)
            print("det3:",np.linalg.det(Info[:3,:3].data.numpy()))
            print("det3:",np.linalg.det(Info[:5,:5].data.numpy()))

    #count parameters
    usedup = int(sum(theta_parts[pname].nelement() for pname in theta_names))

    #sample top-level parameters
    params = pyro.sample('theta',
                    DeltaMVN(thetaMean,
                                precision_matrix=big_hessian),
                    infer={'is_auxiliary': True})

    #decompose theta into specific values
    tmptheta = params
    for pname in theta_names:
        phat = hat_data[pname]
        elems = phat.nelement()
        #print(f"pname, phat: {pname}, {phat}")
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        pdat = transformations[pname](pdat)
        #print(f"adding {pname} from theta ({elems}, {phat.size()})" )

        pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))

    #N of t_part should be left over at end, but we want full_N, so:

    with full_units():
        pyro.sample("t_part", dist.Delta(full_tpart))

    for k,v in fhat_data.items():
        pyro.sample(k, dist.Delta(transformations[k](v)))

    #
    #print("end guide.",theta[:3],mode_hat,nscale_hat,tscale_hat,Info[:5,:5],Info[-3:,-3:])
    #
    #print(".....1....",true_g_hat,theta[-6:])
    #print(".....2....",theta[-9:-6])
    def fix_m_grad():
        if torch.any(torch.isnan(mode_hat.grad)):
            complain( "mode_hat.grad")
        if torch.any(torch.isnan(dm.grad)):
            complain("dm.grad")
        else:
            mode_hat.grad = mode_hat.grad + dm.grad
        mode_hat.grad[mode_hat.grad == float("Inf")] = 1e10
        mode_hat.grad[mode_hat.grad == float("-Inf")] = -1e10
        #print("mode_hat.grad",mode_hat.grad)
    mode_hat.fix_grad = fix_m_grad
    #
    def fix_t_grad():
        if torch.any(torch.isnan(ltscale_hat.grad)):
            complain( "ltscale_hat.grad")
        if torch.any(torch.isnan(dtr.grad)):
            complain( "dtr.grad")
        else:
            ltscale_hat.grad = ltscale_hat.grad + dtr.grad
        ltscale_hat.grad[ltscale_hat.grad == float("Inf")] = 1e10
        ltscale_hat.grad[ltscale_hat.grad == float("-Inf")] = -1e10
        #print("ltscale_hat.grad",ltscale_hat.grad)
    ltscale_hat.fix_grad = fix_t_grad

    def fix_df_grad():
        if torch.any(torch.isnan(ldfraw_hat.grad)):
            complain( "ldfraw_hat.grad")
        if torch.any(torch.isnan(ddfr.grad)):
            complain( "ddfr.grad")
        else:
            ldfraw_hat.grad = ldfraw_hat.grad + ddfr.grad
        ldfraw_hat.grad[ldfraw_hat.grad == float("Inf")] = 1e10
        ldfraw_hat.grad[ldfraw_hat.grad == float("-Inf")] = -1e10
        #print("ldfraw_hat.grad",ldfraw_hat.grad)
    ldfraw_hat.fix_grad = fix_df_grad
    #




    #
def amortized_meanfield(N,full_N,indices,effects,full_effects,errors,full_errors,maxError,weight=1.,scalehyper=ts(4.),tailhyper=ts(10.)):
    #print("guide2 start")
    #
    hat_data = OrderedDict()
    mode_hat = pyro.param("mode_hat",ts(0.))
    hat_data.update(modal_effect=mode_hat)

    nscale_hat = pyro.param("nscale_hat",ts(1.)) #log this? nah; just use "out of box"
    hat_data.update(norm_scale=nscale_hat)

    tscale_hat = pyro.param("tscale_hat",ts(1.))
    hat_data.update(t_scale=tscale_hat)

    normpart, tpart = getMLE(nscale_hat, tscale_hat, errors, effects)

    true_n_hat = normpart * nscale_hat / (nscale_hat + errors)
    hat_data.update(norm_part=true_n_hat)

    true_g_hat = tpart
    hat_data.update(t_part=true_g_hat)

    #Get hessian
    thetaMean = torch.cat([thetaPart.view(-1) for thetaPart in hat_data.values()],0)
    nparams = len(thetaMean)

    theta_scale = pyro.param("theta_scale",torch.ones(nparams))

    #declare global-level psi params
    theta = pyro.sample('theta',
                    dist.Normal(thetaMean,torch.abs(theta_scale)).independent(1),
                    infer={'is_auxiliary': True})

    #decompose theta into specific values
    tmptheta = theta
    for pname, phat in hat_data.items():
        elems = phat.nelement()
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        #print(f"adding {pname} from theta ({elems}, {phat.size()})" )

        pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))
    #

SUBSAMPLE_N = 8

data = pd.read_csv('testresults/effects_errors.csv')
echs_effects = torch.tensor(data.effects)
errors = torch.tensor(data.errors)
N = len(echs_effects)
if False: #smaller
    N = 3
    assert N > SUBSAMPLE_N
    echs_effects = echs_effects[:N]
    errors = errors[:N]
base_scale = 1.
modal_effect = 1.*base_scale
tdom_fat_params = dict(modal_effect=ts(modal_effect),
                            df=ts(-1.),
                            t_scale=ts(2.))
#
ndom_fat_params = dict(modal_effect=ts(modal_effect),
                            df=ts(-1.),
                            t_scale=ts(-2.))
#
tdom_norm_params = dict(modal_effect=ts(modal_effect),
                            df=ts(3.),
                            t_scale=ts(2.))
#
ndom_norm_params = dict(modal_effect=ts(modal_effect),
                            df=ts(3.),
                            t_scale=ts(-2.))
#
#fake_effects = model(N,full_N,indices,echs_effects,errors,fixedParams = tdom_params)
#print("Fake:",fake_effects)

autoguide = AutoDiagonalNormal(model)
guides = OrderedDict(
                    meanfield=autoguide,
                    laplace=laplace,
                    amortized_meanfield = amortized_meanfield,
                    amortized_laplace = amortized_laplace,
                    amortized_deterministic_laplace = amortized_deterministic_laplace
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
    for item in ("mode_hat","ltscale_hat","ldfraw_hat","lpsi",):
        try:
            result.append(store[item])
        except:
            result.append("")

    return result

def getMeanfieldParams():
    store = pyro.get_param_store()
    return list(store["auto_loc"])[:3]

def trainGuide(guidename = "laplace",
            nparticles = 1,
            trueparams = tdom_fat_params,
            filename = None):

    guide = guides[guidename]
    weight = N * 1. / SUBSAMPLE_N
    maxError = torch.max(errors)
    effects = model(N,N,  None,   echs_effects,echs_effects,errors,errors,     maxError,weight,fixedParams = trueparams)
                #N,full_N,indices,effects,     full_effects,errors,full_errors,maxError,weight=1.,scalehyper=ts(4.),tailhyper=ts(10.)):
    print("guidename",guidename)
    print("trueparams",trueparams)
    print("esize",effects.size())

    if filename is None:
        file = FakeSink()
    else:
        file = open(filename,"a")
    writer = csv.writer(file)

    #guide = guide2
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO(nparticles))
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'betas': (0.99,0.9999)}), Trace_ELBO(nparticles)) #.72
    svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'betas': (0.8,0.9)}), Trace_ELBO(nparticles)) #?
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'clip_norm': 5.0}), Trace_ELBO(nparticles)) #.66
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'weight_decay': ...}), Trace_ELBO(nparticles))
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'eps': 1e-5}), Trace_ELBO(nparticles))
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'eps': 1e-10}), Trace_ELBO(nparticles))
    #svi = SVI(model, guide, AdagradRMSProp({}), Trace_ELBO(nparticles))

    pyro.clear_param_store()
    losses = []
    mean_losses = [] #(moving average)
    runtime = time.time()
    base_line = [guidename, runtime,
                    [trueparams[item] for item in ("modal_effect",
                                    "df","t_scale")]]
    for i in range(3001):
        indices = torch.randperm(N)[:SUBSAMPLE_N]
        loss = svi.step(SUBSAMPLE_N,N,indices,
                        effects.index_select(0,indices),effects,
                        errors.index_select(0,indices),errors,
                        maxError,
                        weight,ts(10.),ts(10.))
                    #N,full_N,indices,
                    #effects,     full_effects,
                    #errors,full_errors,
                    #maxError,weight=1.,scalehyper=ts(4.),tailhyper=ts(10.)):

        if len(losses)==0:
            mean_losses.append(loss)
        else:
            mean_losses.append((mean_losses[-1] * 49. + loss) / 50.)
        losses.append(loss)
        if i % 10 == 0:
            try:
                writer.writerow(base_line + [i, time.time(), loss] + getLaplaceParams())
            except:
                writer.writerow(base_line + [i, time.time(), loss] + getMeanfieldParams())
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
    ##

    print("guidename",guidename)
    print("trueparams",trueparams)
    for (key, val) in sorted(pyro.get_param_store().items()):
        print(f"{key}:\n{val}")
