#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

from importlib import reload
import csv
import time
import math

from matplotlib import pyplot as plt
from collections import OrderedDict

import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam, AdagradRMSProp
from pyro import poutine
from pyro.contrib.autoguide import AutoDiagonalNormal
if True:#False:#
    import myhessian
else:
    import hessian as myhessian
import numpy as np
import pandas as pd

from lambertw import lambertw
import go_or_nogo
ts = torch.tensor

pyro.enable_validation(True)
pyro.set_rng_seed(0)



BASE_PSI = .01

EULER_CONSTANT = 0.5772156649015328606065120900824024310421
GUMBEL_SD = math.pi/math.sqrt(6.)



def infoToM(Info,psi=None):
    tlen = len(Info)
    if psi is None:
        psi = torch.ones(tlen) * BASE_PSI
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



def model(N,effects,errors,totvar,
            scalehyper=ts(10.),tailhyper=ts(10.),
            fixedParams = None): #groups, subgroups, groupsize by trial, options

    if fixedParams is not None:
        pass#print("model for fake")
    else:
        pass#print("model for real")

    #prior on μ_τ
    modal_effect = pyro.sample('modal_effect',dist.Normal(ts(0.),ts(20.)))

    #prior on sd(ν)
    norm_scale = pyro.sample('norm_scale',dist.Normal(ts(0.),torch.abs(scalehyper)))

    #prior on sd(γ)
    gum_scale = pyro.sample('gum_scale',dist.Normal(ts(0.),torch.abs(scalehyper)))

    if fixedParams is not None:
        #
        modal_effect = fixedParams['modal_effect']
        norm_scale = fixedParams['norm_scale']
        gum_scale = fixedParams['gum_scale']

    norm_part = pyro.sample('norm_part',dist.Normal(ts(0.).expand(N),ts(1.)).to_event(1))
    gum_part = pyro.sample('gum_part',dist.Gumbel(ts(-EULER_CONSTANT).expand(N),ts(1.)).to_event(1))

    #Latent true values (offset)
    truth = modal_effect + norm_part * torch.abs(norm_scale) + gum_part * gum_scale / GUMBEL_SD

    #Observations conditional on truth (likelihood)
    if fixedParams is not None:
        observations = pyro.sample('observations', dist.Normal(truth,errors).to_event(1))
        print("Not none.")
    else:
        try:
            observations = pyro.sample('observations', dist.Normal(truth,errors).to_event(1), obs=effects)
        except:
            print("ERROR", truth, errors)
            print("ERROR", modal_effect, norm_scale, gum_scale)
            raise

    if fixedParams is not None:
        return observations
    #print("end model",modal_effect,norm_scale,gum_scale,gum_part)


def laplace(N,effects,errors,totvar,scalehyper,tailhyper):
    hat_data = OrderedDict() #
    mode_hat = pyro.param("mode_hat",ts(0.))
    hat_data.update(modal_effect=mode_hat)

    nscale_hat = pyro.param("nscale_hat",ts(1.)) #log this? nah; just use "out of box"
    hat_data.update(norm_scale=nscale_hat)

    gscale_hat = pyro.param("gscale_hat",ts(1.))
    hat_data.update(gum_scale=gscale_hat)

    true_n_hat = pyro.param("true_n_hat",effects/2)
    hat_data.update(norm_part=true_n_hat)

    true_g_hat = pyro.param("true_g_hat",effects/2)
    hat_data.update(gum_part=true_g_hat)

    #Get hessian
    thetaMean = torch.cat([thetaPart.view(-1) for thetaPart in hat_data.values()],0)

    hessCenter = pyro.condition(model,hat_data)
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)(N,effects,errors,totvar,scalehyper,tailhyper) #*args,**kwargs)
    logPosterior = blockedTrace.log_prob_sum()
    Info = -myhessian.hessian(logPosterior, hat_data.values())#, allow_unused=True)
    #print("det:",np.linalg.det(Info.data.numpy()))
    #print(Info)

    if True: #print determinant
        det = np.linalg.det(Info.data.numpy())
        print("det:",det)

        #print("Got hessian")
        if math.isinf(det):
            print("Inf:",Info)
            print("det3:",np.linalg.det(Info[:3,:3].data.numpy()))
            print("det3:",np.linalg.det(Info[:5,:5].data.numpy()))

    #declare global-level psi params
    theta = pyro.sample('theta',
                    dist.MultivariateNormal(thetaMean,
                                precision_matrix=Info + infoToM(Info)),
                    infer={'is_auxiliary': True})

    #decompose theta into specific values
    tmptheta = theta
    for pname, phat in hat_data.items():
        elems = phat.nelement()
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        #print(f"adding {pname} from theta ({elems}, {phat.size()})" )

        pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))
    #


def getMLE(nscale, gscale, error, obs, tol=1e-5, maxit=100):
    sigmasq = nscale**2 + error**2
    varratio = sigmasq/gscale**2
    adjusted_obs = obs + gscale * EULER_CONSTANT

    w = lambertw(varratio * torch.exp(sigmasq/gscale * (1/gscale-adjusted_obs/sigmasq))
                                ,tol=tol, maxit=maxit)
    gumpart_raw = gscale * (w - varratio) + adjusted_obs
    if torch.any(torch.isnan(gumpart_raw)):
        print("getMLE: replacing nans")
        newgum = torch.zeros_like(gumpart_raw)
        full_variance = gscale**2 + sigmasq
        for i in range(len(gumpart_raw)):
            if torch.isnan(gumpart_raw[i]):
                newgum[i] += (adjusted_obs[i] * gscale**2 / full_variance).view(1)
            else:
                newgum[i] += gumpart_raw[i]
        gumpart_raw = newgum
    normpart = (adjusted_obs-gumpart_raw) * nscale / sigmasq #normpart_raw would have gumpart**2
    return (normpart, (gumpart_raw/gscale-EULER_CONSTANT)*GUMBEL_SD)


def getdens(nscale,gscale,np,gp):
    return (dist.Normal(0,nscale).log_prob(np) +
            dist.Gumbel(0,torch.abs(gscale)).log_prob(torch.sign(gscale)*gp))

def testMLE(n=20):
    for i in range(n):
        fac = -torch.log(torch.rand([1]))
        gscale = fac * (1. - 2. * torch.rand([1]))
        nscale = fac - torch.abs(gscale)
        obs = fac * (-torch.log(torch.rand([1]))-1)
        npr, gpr = getMLE(nscale,gscale,0.,obs)
        print(obs,nscale,gscale,npr,gpr)
        ml = getdens(nscale,gscale,npr*nscale,gpr*gscale)
        for delta in [.0001,-.0001]:
            mlnot = getdens(nscale,gscale,(npr+delta)*nscale,(gpr-delta)*gscale)
            if mlnot < ml:
                print(f"getMLE seems to have failed, {fac} {nscale} {gscale} {obs} {npr} {gpr} {delta} {ml} {mlnot}")

                print(fac.item())
                plt.plot([getdens(nscale,gscale,(npr+delta)*nscale,(gpr-delta)*gscale) for delta in np.linspace(-.5*fac.item(), .5*fac.item(), 100)])
                plt.xlabel('dg')
                plt.ylabel('dens')
                plt.show()
                plt.clf()
                #raise "Broken!"

LAMBERT_MAX_ITERS = 10
LAMBERT_TOL = 1e-2

def conditional_normal(full_mean, full_precision, n, first_draw, full_cov=None):
    if full_cov is None:
        full_cov = torch.inverse(full_precision)

    new_precision = full_precision[n:,n:]
    new_mean = full_mean[n:] + torch.mv(torch.mm(full_cov[n:,:n],
                                                torch.inverse(full_cov[:n,:n])),
                                        first_draw - full_mean[:n])
    return(new_mean,new_precision)

def get_unconditional_cov(full_precision, n):
    #TODO: more efficient
    return(torch.inverse(full_precision)[:n,:n])

def amortized_laplace(N,data,errors,totvar,scalehyper,tailhyper):
    hat_data = OrderedDict()
    #this will hold values to condition the model and get the Hessian

    mode_hat = pyro.param("mode_hat",torch.zeros([]))
    hat_data.update(modal_effect=mode_hat)

    nscale_hat = pyro.param("nscale_hat",torch.ones([])) #log this? nah; just use "out of box"
    hat_data.update(norm_scale=nscale_hat)

    gscale_hat = pyro.param("gscale_hat",torch.ones([]))
    hat_data.update(gum_scale=gscale_hat)

    dm = mode_hat.detach().requires_grad_()
    dn = nscale_hat.detach().requires_grad_()
    dg = gscale_hat.detach().requires_grad_()

    normpart, gumpart = getMLE(dn, dg, errors, data-dm,
                            maxit=LAMBERT_MAX_ITERS, tol=LAMBERT_TOL)
    #print("normpart.requires_grad:",normpart.requires_grad)

    true_n_hat = normpart#.detach()
    #true_n_hat.requires_grad = True
    hat_data.update(norm_part=true_n_hat)

    true_g_hat = gumpart#.detach()
    #true_g_hat.requires_grad = True
    hat_data.update(gum_part=true_g_hat)

    #Get hessian
    thetaMean = torch.cat([thetaPart.view(-1) for thetaPart in hat_data.values()],0)

    hessCenter = pyro.condition(model,hat_data)
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)(N,data,errors,totvar,scalehyper,tailhyper) #*args,**kwargs)
    logPosterior = blockedTrace.log_prob_sum()
    theta_names = ["modal_effect", "norm_scale", "gum_scale"]
    theta_parts = dict((theta_name,hat_data[theta_name]) for theta_name in theta_names)
    Info = -myhessian.hessian(logPosterior, hat_data.values())#, allow_unused=True)

    if False: #code for testing hessian grads
        dumb_loss = torch.sum(Info)
        print(f"mode_hat.grad:{mode_hat.grad}; dm.grad:{ dm.grad}")
        dumb_loss.backward(retain_graph=True)
        print(f"mode_hat.grad:{mode_hat.grad}; dm.grad:{ dm.grad}")


    if False: #print determinant
        det = np.linalg.det(Info.data.numpy())
        print("det:",det)

        #print("Got hessian")
        if math.isinf(det):
            print("Inf:",Info)
            print("det3:",np.linalg.det(Info[:3,:3].data.numpy()))
            print("det3:",np.linalg.det(Info[:5,:5].data.numpy()))

    #ensure positive definite
    big_hessian = Info + infoToM(Info)

    #count parameters
    usedup = int(sum(theta_parts[pname].nelement() for pname in theta_names))

    #invert matrix (maybe later, smart)
    theta_cov = get_unconditional_cov(big_hessian,usedup)

    #sample top-level parameters
    theta = pyro.sample('theta',
                    dist.MultivariateNormal(thetaMean[:usedup],
                                theta_cov),
                    infer={'is_auxiliary': True})

    #decompose theta into specific values
    tmptheta = theta
    for pname in theta_names:
        phat = theta_parts[pname]
        elems = phat.nelement()
        #print(f"pname, phat: {pname}, {phat}")
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        #print(f"adding {pname} from theta ({elems}, {phat.size()})" )

        pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))

    #sample unit-level parameters, conditional on top-level ones
    global_indices = ts(range(usedup))
    base_theta = theta
    base_theta_hat = thetaMean[:usedup]
    ylist = []
    for i in range(N):
        #
        #

        precinct_indices = ts([usedup + i, usedup + N + i]) #norm_part[i], gum_part[i]

        full_indices = torch.cat([global_indices, precinct_indices],0)

        full_precision = big_hessian.index_select(0,full_indices).index_select(1,full_indices) #TODO: do in-place?
        full_mean = thetaMean.index_select(0,full_indices) #TODO: do in-place!
        new_mean, new_precision = conditional_normal(full_mean, full_precision, usedup, theta)



        try:
            ylist.append( pyro.sample(f"y_{i}",
                            dist.MultivariateNormal(new_mean, precision_matrix=new_precision),
                            infer={'is_auxiliary': True}))
        except:
            print(new_precision)
            print(f"det:{np.linalg.det(new_precision.data.numpy())}")
            raise
    norm_part = torch.cat([y[0].view(-1) for y in ylist],0)
    gum_part = torch.cat([y[1].view(-1) for y in ylist],0)
    pyro.sample("norm_part", dist.Delta(norm_part).to_event(1))
    pyro.sample("gum_part", dist.Delta(gum_part).to_event(1))
    #
    #print("end guide.",theta[:3],mode_hat,nscale_hat,gscale_hat,Info[:5,:5],Info[-3:,-3:])
    #
    #print(".....1....",true_g_hat,theta[-6:])
    #print(".....2....",theta[-9:-6])
    def fix_m_grad():
        mode_hat.grad = mode_hat.grad + dm.grad
    mode_hat.fix_grad = fix_m_grad
    #
    def fix_n_grad():
        nscale_hat.grad = nscale_hat.grad + dn.grad
    nscale_hat.fix_grad = fix_n_grad
    #
    def fix_g_grad():
        gscale_hat.grad = gscale_hat.grad + dg.grad
    gscale_hat.fix_grad = fix_g_grad
    #


def amortized_laplace(N,data,errors,totvar,scalehyper,tailhyper):
    hat_data = OrderedDict()
    #this will hold values to condition the model and get the Hessian

    mode_hat = pyro.param("mode_hat",torch.zeros([]))
    hat_data.update(modal_effect=mode_hat)

    nscale_hat = pyro.param("nscale_hat",torch.ones([])) #log this? nah; just use "out of box"
    hat_data.update(norm_scale=nscale_hat)

    gscale_hat = pyro.param("gscale_hat",torch.ones([]))

    hat_data.update(gum_scale=gscale_hat)

    dm = mode_hat.detach().requires_grad_()
    dn = nscale_hat.detach().requires_grad_()
    dg = gscale_hat.detach().requires_grad_()

    normpart, gumpart = getMLE(dn, dg, errors, data-dm,
                            maxit=LAMBERT_MAX_ITERS, tol=LAMBERT_TOL)
    #print("normpart.requires_grad:",normpart.requires_grad)

    true_n_hat = normpart#.detach()
    #true_n_hat.requires_grad = True
    hat_data.update(norm_part=true_n_hat)

    true_g_hat = gumpart#.detach()
    #true_g_hat.requires_grad = True
    hat_data.update(gum_part=true_g_hat)

    #Get hessian
    thetaMean = torch.cat([thetaPart.view(-1) for thetaPart in hat_data.values()],0)

    hessCenter = pyro.condition(model,hat_data)
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)(N,data,errors,totvar,scalehyper,tailhyper) #*args,**kwargs)
    logPosterior = blockedTrace.log_prob_sum()
    theta_names = ["modal_effect", "norm_scale", "gum_scale"]
    theta_parts = dict((theta_name,hat_data[theta_name]) for theta_name in theta_names)
    Info = -myhessian.hessian(logPosterior, hat_data.values())#, allow_unused=True)

    if False: #code for testing hessian grads
        dumb_loss = torch.sum(Info)
        print(f"mode_hat.grad:{mode_hat.grad}; dm.grad:{ dm.grad}")
        dumb_loss.backward(retain_graph=True)
        print(f"mode_hat.grad:{mode_hat.grad}; dm.grad:{ dm.grad}")


    if False: #print determinant
        det = np.linalg.det(Info.data.numpy())
        print("det:",det)

        #print("Got hessian")
        if math.isinf(det):
            print("Inf:",Info)
            print("det3:",np.linalg.det(Info[:3,:3].data.numpy()))
            print("det3:",np.linalg.det(Info[:5,:5].data.numpy()))

    #ensure positive definite
    big_hessian = Info + infoToM(Info)

    #count parameters
    usedup = int(sum(theta_parts[pname].nelement() for pname in theta_names))

    #invert matrix (maybe later, smart)
    theta_cov = get_unconditional_cov(big_hessian,usedup)

    #sample top-level parameters
    theta = pyro.sample('theta',
                    dist.MultivariateNormal(thetaMean[:usedup],
                                theta_cov),
                    infer={'is_auxiliary': True})

    #decompose theta into specific values
    tmptheta = theta
    for pname in theta_names:
        phat = theta_parts[pname]
        elems = phat.nelement()
        #print(f"pname, phat: {pname}, {phat}")
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        #print(f"adding {pname} from theta ({elems}, {phat.size()})" )

        pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))

    #sample unit-level parameters, conditional on top-level ones
    global_indices = ts(range(usedup))
    base_theta = theta
    base_theta_hat = thetaMean[:usedup]
    ylist = []
    for i in range(N):
        #
        #

        precinct_indices = ts([usedup + i, usedup + N + i]) #norm_part[i], gum_part[i]

        full_indices = torch.cat([global_indices, precinct_indices],0)

        full_precision = big_hessian.index_select(0,full_indices).index_select(1,full_indices) #TODO: do in-place?
        full_mean = thetaMean.index_select(0,full_indices) #TODO: do in-place!
        new_mean, new_precision = conditional_normal(full_mean, full_precision, usedup, theta)



        try:
            ylist.append( pyro.sample(f"y_{i}",
                            dist.MultivariateNormal(new_mean, precision_matrix=new_precision),
                            infer={'is_auxiliary': True}))
        except:
            print(new_precision)
            print(f"det:{np.linalg.det(new_precision.data.numpy())}")
            raise
    norm_part = torch.cat([y[0].view(-1) for y in ylist],0)
    gum_part = torch.cat([y[1].view(-1) for y in ylist],0)
    pyro.sample("norm_part", dist.Delta(norm_part).to_event(1))
    pyro.sample("gum_part", dist.Delta(gum_part).to_event(1))
    #
    #print("end guide.",theta[:3],mode_hat,nscale_hat,gscale_hat,Info[:5,:5],Info[-3:,-3:])
    #
    #print(".....1....",true_g_hat,theta[-6:])
    #print(".....2....",theta[-9:-6])
    def fix_m_grad():
        mode_hat.grad = mode_hat.grad + dm.grad
    mode_hat.fix_grad = fix_m_grad
    #
    def fix_n_grad():
        nscale_hat.grad = nscale_hat.grad + dn.grad
    nscale_hat.fix_grad = fix_n_grad
    #
    def fix_g_grad():
        gscale_hat.grad = gscale_hat.grad + dg.grad
    gscale_hat.fix_grad = fix_g_grad
    #

def amortized_laplace_narrowed(N,data,errors,totvar,scalehyper,tailhyper):
    hat_data = OrderedDict()
    #this will hold values to condition the model and get the Hessian

    mode_hat = pyro.param("mode_hat",torch.zeros([]))
    hat_data.update(modal_effect=mode_hat)

    nscale_hat = pyro.param("nscale_hat",torch.ones([])) #log this? nah; just use "out of box"
    hat_data.update(norm_scale=nscale_hat)

    gscale_hat = pyro.param("gscale_hat",torch.ones([]))
    hat_data.update(gum_scale=gscale_hat)

    #narrower = pyro.param("narrower",torch.zeros(2)+.001,constraints.positive)

    dm = mode_hat.detach().requires_grad_()
    dn = nscale_hat.detach().requires_grad_()
    dg = gscale_hat.detach().requires_grad_()

    normpart, gumpart = getMLE(dn, dg, errors, data-dm,
                            maxit=LAMBERT_MAX_ITERS, tol=LAMBERT_TOL)
    #print("normpart.requires_grad:",normpart.requires_grad)

    true_n_hat = normpart#.detach()
    #true_n_hat.requires_grad = True
    hat_data.update(norm_part=true_n_hat)

    true_g_hat = gumpart#.detach()
    #true_g_hat.requires_grad = True
    hat_data.update(gum_part=true_g_hat)

    #Get hessian
    thetaMean = torch.cat([thetaPart.view(-1) for thetaPart in hat_data.values()],0)

    hessCenter = pyro.condition(model,hat_data)
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)(N,data,errors,totvar,scalehyper,tailhyper) #*args,**kwargs)
    logPosterior = blockedTrace.log_prob_sum()
    theta_names = ["modal_effect", "norm_scale", "gum_scale"]
    theta_parts = dict((theta_name,hat_data[theta_name]) for theta_name in theta_names)
    Info = -myhessian.hessian(logPosterior, hat_data.values())#, allow_unused=True)

    if False: #code for testing hessian grads
        dumb_loss = torch.sum(Info)
        print(f"mode_hat.grad:{mode_hat.grad}; dm.grad:{ dm.grad}")
        dumb_loss.backward(retain_graph=True)
        print(f"mode_hat.grad:{mode_hat.grad}; dm.grad:{ dm.grad}")


    if False: #print determinant
        det = np.linalg.det(Info.data.numpy())
        print("det:",det)

        #print("Got hessian")
        if math.isinf(det):
            print("Inf:",Info)
            print("det3:",np.linalg.det(Info[:3,:3].data.numpy()))
            print("det3:",np.linalg.det(Info[:5,:5].data.numpy()))

    #ensure positive definite
    big_hessian = (Info + infoToM(Info) * #Shur/elementwise multiplication, not matrix multiplication
                    torch.diag(torch.cat(
                        [torch.ones(3),
                        torch.ones(N)*5.,
                        torch.ones(N)*5.,
                        ]
                    ,0)))


    #count parameters
    usedup = int(sum(theta_parts[pname].nelement() for pname in theta_names))

    #invert matrix (maybe later, smart)
    theta_cov = get_unconditional_cov(big_hessian,usedup)

    #sample top-level parameters
    theta = pyro.sample('theta',
                    dist.MultivariateNormal(thetaMean[:usedup],
                                theta_cov),
                    infer={'is_auxiliary': True})

    #decompose theta into specific values
    tmptheta = theta
    for pname in theta_names:
        phat = theta_parts[pname]
        elems = phat.nelement()
        #print(f"pname, phat: {pname}, {phat}")
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        #print(f"adding {pname} from theta ({elems}, {phat.size()})" )

        pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))

    #sample unit-level parameters, conditional on top-level ones
    global_indices = ts(range(usedup))
    base_theta = theta
    base_theta_hat = thetaMean[:usedup]
    ylist = []
    for i in range(N):
        #
        #

        precinct_indices = ts([usedup + i, usedup + N + i]) #norm_part[i], gum_part[i]

        full_indices = torch.cat([global_indices, precinct_indices],0)

        full_precision = big_hessian.index_select(0,full_indices).index_select(1,full_indices) #TODO: do in-place?
        full_mean = thetaMean.index_select(0,full_indices) #TODO: do in-place!
        new_mean, new_precision = conditional_normal(full_mean, full_precision, usedup, theta)



        try:
            ylist.append( pyro.sample(f"y_{i}",
                            dist.MultivariateNormal(new_mean, precision_matrix=new_precision),
                            infer={'is_auxiliary': True}))
        except:
            print(new_precision)
            print(f"det:{np.linalg.det(new_precision.data.numpy())}")
            raise
    norm_part = torch.cat([y[0].view(-1) for y in ylist],0)
    gum_part = torch.cat([y[1].view(-1) for y in ylist],0)
    pyro.sample("norm_part", dist.Delta(norm_part).to_event(1))
    pyro.sample("gum_part", dist.Delta(gum_part).to_event(1))
    #
    #print("end guide.",theta[:3],mode_hat,nscale_hat,gscale_hat,Info[:5,:5],Info[-3:,-3:])
    #
    #print(".....1....",true_g_hat,theta[-6:])
    #print(".....2....",theta[-9:-6])
    def fix_m_grad():
        mode_hat.grad = mode_hat.grad + dm.grad
    mode_hat.fix_grad = fix_m_grad
    #
    def fix_n_grad():
        nscale_hat.grad = nscale_hat.grad + dn.grad
    nscale_hat.fix_grad = fix_n_grad
    #
    def fix_g_grad():
        gscale_hat.grad = gscale_hat.grad + dg.grad
    gscale_hat.fix_grad = fix_g_grad
    #

def amortized_laplace_reduced(N,data,errors,totvar,scalehyper,tailhyper):
    hat_data = OrderedDict()
    #this will hold values to condition the model and get the Hessian

    mode_hat = pyro.param("mode_hat",torch.zeros([]))
    hat_data.update(modal_effect=mode_hat)


    gscale_hat = pyro.param("gscale_hat",torch.ones([]))
    hat_data.update(gum_scale=gscale_hat)

    nscale_hat = totvar - gscale_hat**2
    hat_data.update(norm_scale=nscale_hat)

    dm = mode_hat.detach().requires_grad_()
    dg = gscale_hat.detach().requires_grad_()
    dn = totvar - dg**2

    normpart, gumpart = getMLE(dn, dg, errors, data-dm,
                            maxit=LAMBERT_MAX_ITERS, tol=LAMBERT_TOL)
    #print("normpart.requires_grad:",normpart.requires_grad)

    true_n_hat = normpart#.detach()
    #true_n_hat.requires_grad = True
    hat_data.update(norm_part=true_n_hat)

    true_g_hat = gumpart#.detach()
    #true_g_hat.requires_grad = True
    hat_data.update(gum_part=true_g_hat)

    #Get hessian
    thetaMean = torch.cat([thetaPart.view(-1) for thetaPart in hat_data.values()],0)

    hessCenter = pyro.condition(model,hat_data)
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)(N,data,errors,totvar,scalehyper,tailhyper) #*args,**kwargs)
    logPosterior = blockedTrace.log_prob_sum()
    theta_names = ["modal_effect", "gum_scale"]
    theta_parts = dict((theta_name,hat_data[theta_name]) for theta_name in theta_names)
    Info = -myhessian.hessian(logPosterior, hat_data.values())#, allow_unused=True)

    if False: #code for testing hessian grads
        dumb_loss = torch.sum(Info)
        print(f"mode_hat.grad:{mode_hat.grad}; dm.grad:{ dm.grad}")
        dumb_loss.backward(retain_graph=True)
        print(f"mode_hat.grad:{mode_hat.grad}; dm.grad:{ dm.grad}")


    if False: #print determinant
        det = np.linalg.det(Info.data.numpy())
        print("det:",det)

        #print("Got hessian")
        if math.isinf(det):
            print("Inf:",Info)
            print("det3:",np.linalg.det(Info[:3,:3].data.numpy()))
            print("det3:",np.linalg.det(Info[:5,:5].data.numpy()))

    #ensure positive definite
    big_hessian = Info + infoToM(Info)

    #count parameters
    usedup = int(sum(theta_parts[pname].nelement() for pname in theta_names))

    usedup2 = usedup + 1 #norm_scale

    #invert matrix (maybe later, smart)
    theta_cov = get_unconditional_cov(big_hessian,usedup)

    #sample top-level parameters
    theta = pyro.sample('theta',
                    dist.MultivariateNormal(thetaMean[:usedup],
                                theta_cov),
                    infer={'is_auxiliary': True})

    #decompose theta into specific values
    tmptheta = theta
    for pname in theta_names:
        phat = theta_parts[pname]
        elems = phat.nelement()
        #print(f"pname, phat: {pname}, {phat}")
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        #print(f"adding {pname} from theta ({elems}, {phat.size()})" )

        pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))

    #sample unit-level parameters, conditional on top-level ones
    global_indices = ts(range(usedup))
    base_theta = theta
    base_theta_hat = thetaMean[:usedup]
    ylist = []
    for i in range(N):
        #
        #

        precinct_indices = ts([usedup2 + i, usedup2 + N + i]) #norm_part[i], gum_part[i]

        full_indices = torch.cat([global_indices, precinct_indices],0)

        full_precision = big_hessian.index_select(0,full_indices).index_select(1,full_indices) #TODO: do in-place?
        full_mean = thetaMean.index_select(0,full_indices) #TODO: do in-place!
        new_mean, new_precision = conditional_normal(full_mean, full_precision, usedup, theta)



        try:
            ylist.append( pyro.sample(f"y_{i}",
                            dist.MultivariateNormal(new_mean, precision_matrix=new_precision),
                            infer={'is_auxiliary': True}))
        except:
            print(new_precision)
            print(f"det:{np.linalg.det(new_precision.data.numpy())}")
            raise
    norm_part = torch.cat([y[0].view(-1) for y in ylist],0)
    gum_part = torch.cat([y[1].view(-1) for y in ylist],0)
    pyro.sample("norm_part", dist.Delta(norm_part).to_event(1))
    pyro.sample("gum_part", dist.Delta(gum_part).to_event(1))
    #
    #print("end guide.",theta[:3],mode_hat,nscale_hat,gscale_hat,Info[:5,:5],Info[-3:,-3:])
    #
    #print(".....1....",true_g_hat,theta[-6:])
    #print(".....2....",theta[-9:-6])
    def fix_m_grad():
        mode_hat.grad = mode_hat.grad + dm.grad
    mode_hat.fix_grad = fix_m_grad
    #
    def fix_g_grad():
        gscale_hat.grad = gscale_hat.grad + dg.grad
    gscale_hat.fix_grad = fix_g_grad
    #


    #
def amortized_meanfield(N,effects,errors,totvar,scalehyper,tailhyper):
    #print("guide2 start")
    #
    hat_data = OrderedDict()
    mode_hat = pyro.param("mode_hat",ts(0.))
    hat_data.update(modal_effect=mode_hat)

    nscale_hat = pyro.param("nscale_hat",ts(1.)) #log this? nah; just use "out of box"
    hat_data.update(norm_scale=nscale_hat)

    gscale_hat = pyro.param("gscale_hat",ts(1.))
    hat_data.update(gum_scale=gscale_hat)

    normpart, gumpart = getMLE(nscale_hat, gscale_hat, errors, effects)

    true_n_hat = normpart * nscale_hat / (nscale_hat + errors)
    hat_data.update(norm_part=true_n_hat)

    true_g_hat = gumpart
    hat_data.update(gum_part=true_g_hat)

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



data = pd.read_csv('testresults/effects_errors.csv')
echs_effects = torch.tensor(data.effects)
errors = torch.tensor(data.errors)
N = len(echs_effects)
base_scale = 10
modal_effect = -.005*base_scale
gdom_params = dict(modal_effect=ts(modal_effect),
                            norm_scale=ts(.01*base_scale),
                            gum_scale=ts(.2*base_scale))
#
ndom_params = dict(modal_effect=ts(modal_effect),
                            norm_scale=ts(.2*base_scale),
                            gum_scale=ts(.01*base_scale))
#
even_params = dict(modal_effect=ts(modal_effect),
                            norm_scale=ts(.1*base_scale),
                            gum_scale=ts(.1*base_scale))
#
neggum_params = dict(modal_effect=ts(modal_effect),
                            norm_scale=ts(.01*base_scale),
                            gum_scale=ts(-.2*base_scale))
#
#fake_effects = model(N,echs_effects,errors,fixedParams = gdom_params)
#print("Fake:",fake_effects)

autoguide = AutoDiagonalNormal(model)
guides = OrderedDict(
                    meanfield=autoguide,
                    laplace=laplace,
                    amortized_meanfield = amortized_meanfield,
                    amortized_laplace = amortized_laplace,
                    amortized_laplace_reduced = amortized_laplace_reduced,
                    amortized_laplace_narrowed = amortized_laplace_narrowed,
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
    for item in ("mode_hat","nscale_hat","gscale_hat","narrower"):
        try:
            result.append(store[item])
        except:
            pass

    return result

def getMeanfieldParams():
    store = pyro.get_param_store()
    return list(store["auto_loc"])[:3]

def trainGuide(guidename = "laplace",
            nparticles = 1,
            trueparams = gdom_params,
            filename = None):

    guide = guides[guidename]
    totvar = trueparams["norm_scale"]**2 + trueparams["gum_scale"]**2
    effects = model(N,echs_effects,errors,totvar,fixedParams = trueparams)

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
                                    "norm_scale","gum_scale")]]
    for i in range(3001):
        loss = svi.step(N,effects,errors,totvar,ts(10.),ts(10.))
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


    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    ##

    for (key, val) in sorted(pyro.get_param_store().items()):
        print(f"{key}:\n{val}")
