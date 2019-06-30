#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

from importlib import reload
import csv
import time
import math
import copy
from statistics import mean

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

from polytopize import approx_eq
from lambertw import lambertw
import go_or_nogo
from draw_ellipses import confidence_ellipse
ts = torch.tensor

pyro.enable_validation(True)
pyro.set_rng_seed(2)



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



def model(obs=ts(0.),df=ts(3.),sig=ts(.4),**kwargs):
    effects = pyro.sample('effects',dist.StudentT(df,torch.zeros(2),
                                                ts(1.)).to_event(1))
    observed = pyro.sample('observed',dist.Normal(torch.sum(effects),sig),obs=obs)


def laplace(*args,assymetry=2.,store=None,**kwargs):
    hat_data = OrderedDict() #
    eff_hat = pyro.param("eff_hat",ts([-assymetry,assymetry]))
    hat_data.update(effects=eff_hat)

    psi = pyro.param("psi",torch.ones(2)*BASE_PSI,constraints.positive)

    if store is not None:
        store_row = dict()
        store.append(store_row)

    #Get hessian
    thetaMean = torch.cat([thetaPart.view(-1) for thetaPart in hat_data.values()],0)

    hessCenter = pyro.condition(model,hat_data)
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)(*args,**kwargs)
    logPosterior = blockedTrace.log_prob_sum()
    Info = -myhessian.hessian(logPosterior, hat_data.values())#, allow_unused=True)
    #print("det:",np.linalg.det(Info.data.numpy()))
    #print(Info)

    #declare global-level psi params
    cov = torch.inverse(Info + infoToM(Info,psi))
    theta = pyro.sample('theta',
                    dist.OMTMultivariateNormal(thetaMean,
                                torch.cholesky(cov)),#.to_event(1),
                    infer={'is_auxiliary': True})

    #decompose theta into specific values
    tmptheta = theta
    for pname, phat in hat_data.items():
        elems = phat.nelement()
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        #print(f"adding {pname} from theta ({elems}, {phat.size()})" )

        pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))
        if store is not None:
            store_row[pname+"_samp"] = pdat

    if store is not None:
        store_row.update(copy.deepcopy(hat_data))
        store_row.update(cov=cov)
    #


def meanfield(*args,assymetry=2.,store=None,**kwargs):
    hat_data = OrderedDict() #
    eff_hat = pyro.param("eff_hat",ts([-assymetry,assymetry]))
    sig = pyro.param("sig",torch.ones(2),constraints.positive)
    hat_data.update(effects=eff_hat,var=sig,cov=torch.diag(sig**2))

    if store is not None:
        store_row = dict()
        store.append(store_row)

    effects = pyro.sample('effects',
                    dist.Normal(eff_hat,
                                hat_data["var"]).to_event(1))


    if store is not None:
        store_row.update([(k,ts(v)) for k,v in hat_data.items()])
    #


def getPosteriorSlice(perp,obs,**kwargs):
    """
    Return (point,logdensity) where point is (x,y), the MLE s.t. y-x=perp;
    and logdensity is the log posterior density at that point.
    """
    sumval = obs.detach().view(1).requires_grad_()

    effects = torch.cat([((sumval-perp)/2).view(1), ((sumval+perp)/2).view(1)],0)

    hessCenter = pyro.condition(model,dict(effects=effects))
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)(obs=obs,**kwargs)
    logPosterior = blockedTrace.log_prob_sum()
    hess,grad = myhessian.hessian(logPosterior, [sumval],
                return_grad=True,)#, allow_unused=True)

    #assert approx_eq(-grad,kwargs["sig"]**2) #missing a factor of sqrt(2), I'll fix this empirically

    distance = grad / -hess[0] #one-dimensional
    #print("getPosteriorSlice",float(logPosterior), float(grad), float(distance),float(hess))
    return (torch.cat([(obs+distance-perp)/2, (obs+distance+perp)/2],0),
            logPosterior + grad * distance / 2)

def getPosteriorSlices(cutoff,delta,obs,**kwargs):
    maxLogDens = curLogDens = -1e6
    slices = []
    perp = -delta / 2
    while curLogDens - maxLogDens > cutoff:
        perp += delta
        pt, curLogDens = getPosteriorSlice(perp,obs=obs,**kwargs)
        if curLogDens > maxLogDens:
            maxLogDens = curLogDens
        slices.append((perp,pt,curLogDens))
    return [(perp,pt,logDens-maxLogDens) for perp,pt,logDens in slices]

def getMeanDens(densTens):
    return torch.mean(torch.exp(densTens))

def getPropInside(minDens, slices,sig):
    densTens = torch.cat([logDens.view(1) for perp, pt, logDens in slices], 0)
    logMeanDens = torch.log(getMeanDens(densTens))
    parts = []
    return torch.mean(torch.exp(densTens-logMeanDens) *
                        (2 * dist.Normal(0.,1.).cdf(
                            torch.sqrt(
                                torch.max(torch.zeros(len(densTens)),
                                    (densTens - minDens) * 2)))
                        - 1))

def getMinDens(alpha, slices, sig):
    minDens = ts(-1., requires_grad=True)
    optimizer = torch.optim.SGD([minDens],lr=5e-1)
    for i in range(20):
        optimizer.zero_grad()
        loss = (alpha - getPropInside(minDens, slices, sig)) ** 2
        #print("backward",loss)
        loss.backward(retain_graph=True)
        optimizer.step()
    #print("minDens",minDens)
    return minDens

def getConfBounds(alpha,slices,**kwargs):
    sig = kwargs["sig"]
    obs = kwargs["obs"]
    minDens = getMinDens(alpha,slices,sig)
    northBorder = []
    for i, (perp, (x,y), logDens) in enumerate(slices):
        if logDens > minDens:
            blockdistance = sig * torch.sqrt((logDens - minDens)/2) # would be *2 inside sqrt but then divide again for euclid
            northBorder.append((ts([x,y]), ts([x+blockdistance,y+blockdistance])))
        else:
            pass#print("getConfBounds??",i,float(logDens),float(minDens),len(northBorder))

    #print("getConfBounds",len(northBorder))#,[float(logDens) for perp, (x,y), logDens in slices])
    swap = ts([1,0])
    upperPart = ([pt for base,pt in northBorder] + #north
                  [2 * base - pt for base,pt in reversed(northBorder)]) #west
    lowerPart = ([(2 * base - pt).index_select(0,swap) for base,pt in northBorder] + #south
                  [pt.index_select(0,swap) for base,pt in reversed(northBorder)]) #east
    if slices[0][2] > minDens: #connected
        fullBorders = [upperPart + lowerPart]
    else: #separate
        fullBorders = [upperPart, lowerPart]
    return fullBorders

def getTruePosteriors(alphas,cutoff,delta,**kwargs):
    slices = getPosteriorSlices(cutoff,delta,**kwargs)
    #print("slices",len(slices))
    curves = []
    for alpha in alphas:
        curves.extend(getConfBounds(alpha,slices,**kwargs))
    return curves

#

def MLE(*args,assymetry=2.,store=None,**kwargs):
    hat_data = OrderedDict() #
    eff_hat = pyro.param("eff_hat",ts([-assymetry,assymetry]))
    hat_data.update(effects=eff_hat,cov=torch.eye(2) * .001)

    if store is not None:
        store_row = dict()
        store.append(store_row)

    effects = pyro.sample('effects',
                    dist.Delta(eff_hat).to_event(1))


    if store is not None:
        store_row.update([(k,ts(v)) for k,v in hat_data.items()])
    #

def getMLE(nscale, gscale, error, obs, tol=1e-5, maxit=100):
    pass
    # sigmasq = nscale**2 + error**2
    # varratio = sigmasq/gscale**2
    # adjusted_obs = obs + gscale * EULER_CONSTANT
    #
    # w = lambertw(varratio * torch.exp(sigmasq/gscale * (1/gscale-adjusted_obs/sigmasq))
    #                             ,tol=tol, maxit=maxit)
    # gumpart_raw = gscale * (w - varratio) + adjusted_obs
    # if torch.any(torch.isnan(gumpart_raw)):
    #     print("getMLE: replacing nans")
    #     newgum = torch.zeros_like(gumpart_raw)
    #     full_variance = gscale**2 + sigmasq
    #     for i in range(len(gumpart_raw)):
    #         if torch.isnan(gumpart_raw[i]):
    #             newgum[i] += adjusted_obs[i] * gscale**2 / full_variance
    #         else:
    #             newgum[i] += gumpart_raw[i]
    #     gumpart_raw = newgum
    # normpart = (adjusted_obs-gumpart_raw) * nscale / sigmasq #normpart_raw would have gumpart**2
    # return (normpart, (gumpart_raw/gscale-EULER_CONSTANT)*GUMBEL_SD)


def getdens(nscale,gscale,np,gp):
    pass

def testMLE(n=20):
    pass

LAMBERT_MAX_ITERS = 10
LAMBERT_TOL = 1e-2
#
# def conditional_normal(full_mean, full_precision, n, first_draw, full_cov=None):
#     if full_cov is None:
#         full_cov = torch.inverse(full_precision)
#
#     new_precision = full_precision[n:,n:]
#     new_mean = full_mean[n:] + torch.mv(torch.mm(full_cov[n:,:n],
#                                                 torch.inverse(full_cov[:n,:n])),
#                                         first_draw - full_mean[:n])
#     return(new_mean,new_precision)
#
# def get_unconditional_cov(full_precision, n):
#     #TODO: more efficient
#     return(torch.inverse(full_precision)[:n,:n])
#
# def amortized_laplace(N,data,errors,totvar,scalehyper,tailhyper)
#   ....

#
#fake_effects = model(N,echs_effects,errors,fixedParams = gdom_params)
#print("Fake:",fake_effects)

autoguide = AutoDiagonalNormal(model)
guides = OrderedDict(
                    meanfield=meanfield,
                    laplace=laplace,
                    MLE=MLE,
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
    for item in ("eff_hat"):
        try:
            result.append(store[item])
        except:
            pass

    return result

def getMeanfieldParams():
    store = pyro.get_param_store()
    return list(store["auto_loc"])[:3]

GAP_BETWEEN_STORE = 15
STORES_BETWEEN_ELLIPSES = 30

LOWDF0 = OrderedDict(obs=ts(0.),df=ts(2.),sig=ts(.4),assymmetry=2.)
LOWDFM = OrderedDict(obs=ts(3.),df=ts(2.),sig=ts(.4),assymmetry=2.)
LOWDFP = OrderedDict(obs=ts(7.),df=ts(2.),sig=ts(.4),assymmetry=2.)
MIDDFP = OrderedDict(obs=ts(4.),df=ts(5.),sig=ts(.4),assymmetry=2.)
HIDF0 = OrderedDict(obs=ts(0.),df=ts(30.),sig=ts(.4),assymmetry=2.)
HIDFP = OrderedDict(obs=ts(3.),df=ts(30.),sig=ts(.4),assymmetry=2.)

def trainGuides(guidenames = ["laplace"],
            nparticles = 1,
            vals = LOWDF0,
            filename = None):

    stores = []
    for guidename in guidenames:
        print(guidename,"vals",vals)
        guide = guides[guidename]

        if filename is None:
            file = FakeSink()
        else:
            file = open(filename,"a")
        writer = csv.writer(file)

        #guide = guide2
        #svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO(nparticles))
        svi = SVI(model, guide, ClippedAdam({'lr': 0.1}), Trace_ELBO(nparticles))
        #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'betas': (0.99,0.9999)}), Trace_ELBO(nparticles)) #.72
        #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'betas': (0.8,0.9)}), Trace_ELBO(nparticles)) #?
        #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'clip_norm': 2.0}), Trace_ELBO(nparticles)) #.66
        #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'weight_decay': ...}), Trace_ELBO(nparticles))
        #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'eps': 1e-5}), Trace_ELBO(nparticles))
        #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'eps': 1e-10}), Trace_ELBO(nparticles))
        #svi = SVI(model, guide, AdagradRMSProp({}), Trace_ELBO(nparticles))

        pyro.clear_param_store()
        losses = []
        mean_losses = [] #(moving average)
        runtime = time.time()
        base_line = [guidename, runtime,] + list(vals.values())
        stores.append([])
        for i in range(3001):
            if i % GAP_BETWEEN_STORE == 0:
                this_store = stores[-1]
            else:
                this_store = None
            loss = svi.step(store=this_store,**vals)
            if len(losses)==0:
                mean_losses.append(loss)
            else:
                mean_losses.append((mean_losses[-1] * 349. + loss) / 350.)
            losses.append(loss)
            if i % 10 == 0:
                try:
                    writer.writerow(base_line + [i, time.time(), loss] + getLaplaceParams())
                except:
                    raise
                    writer.writerow(base_line + [i, time.time(), loss] + getMeanfieldParams())
                reload(go_or_nogo)
                go_or_nogo.demoprintstuff(i,loss)
                try:
                    if mean_losses[-1] > mean_losses[-600] - .01:
                        break
                except:
                    pass
                if go_or_nogo.go:
                    pass
                else:
                    break

        ##
        loss = svi.step(store=this_store,**vals)
        for (key, val) in sorted(pyro.get_param_store().items()):
            print(f"{key}:\n{val}")

        plt.plot(losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))


    ax.set_title('Laplace ellipses')
    colorlist = [(1.,0.,0.),(0.,0.,1.),(0.,1.,0.)] #['red','blue','green','pink']
    for store,color,guidename in zip(stores,colorlist,guidenames):
        for i,line in enumerate(store):
            #print(line)
            if i+1==len(store):
                alpha = .5
                edgecolor = 'black'
                label = guidename
            elif (i % STORES_BETWEEN_ELLIPSES == 0):
                alpha = .003
                edgecolor = None
                label = None
            else:
                alpha = .001
                edgecolor = None
                label = None

            confidence_ellipse(line["effects"],line["cov"],ax,
                    alpha=alpha, facecolor=color, edgecolor=edgecolor, zorder=0,
                    label=label)
            ax.scatter(line["effects"][0].detach().view(1), line["effects"][1].detach().view(1), s=0.5, alpha=.05,color=color)

    lims = [-3,9]
    ax.scatter(lims, lims, s=0.5,alpha=.001)
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)
    ax.legend()


    ps = getTruePosteriors([.5,.95],-9,.1,**vals)


    for p in ps:
        ax.fill([x for x,y in p], [y for x,y in p], fill=False, color=(0.,1.,0.),
                label="True posterior (95%, 50% credible)")
    plt.show()

    ##
