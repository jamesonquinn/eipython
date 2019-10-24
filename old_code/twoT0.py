#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

from importlib import reload
import os
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
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
import numpy as np
import pandas as pd

if True:#False:#
    from utilities import myhessian
else:
    import hessian as myhessian
from utilities.polytopize import approx_eq
from utilities.lambertw import lambertw
from utilities import go_or_nogo
from utilities.draw_ellipses import confidence_ellipse
from utilities.posdef import *
ts = torch.tensor

pyro.enable_validation(True)
pyro.set_rng_seed(2)




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
    cov = torch.inverse(rescaledSDD(Info,psi))
    def doSample(anneal=1.):
        results = OrderedDict()
        thetaDist = dist.OMTMultivariateNormal(thetaMean,
                    torch.cholesky(cov))
        if anneal > 1:
            thetaSamplingDist = dist.OMTMultivariateNormal(thetaMean,
                        torch.cholesky(cov * anneal))
        else:
            thetaSamplingDist = thetaDist
        theta = pyro.sample('theta',
                        thetaSamplingDist,#.to_event(1),
                        infer={'is_auxiliary': True})
        results.update(lpdf=thetaDist.log_prob(theta))

        #decompose theta into specific values
        tmptheta = theta
        for pname, phat in hat_data.items():
            elems = phat.nelement()
            pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
            #print(f"adding {pname} from theta ({elems}, {phat.size()})" )

            result = pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))
            results[pname] = result
            if store is not None:
                store_row[pname+"_samp"] = pdat

        if store is not None:
            store_row.update(copy.deepcopy(hat_data))
            store_row.update(cov=cov)
        return results
    #
    doSample()
    return (eff_hat, cov, doSample)


def meanfield(*args,assymetry=2.,store=None,**kwargs):
    hat_data = OrderedDict() #
    eff_hat = pyro.param("eff_hat",ts([-assymetry,assymetry]))
    sig = pyro.param("sig",torch.ones(2),constraints.positive)
    hat_data.update(effects=eff_hat,var=sig,cov=torch.diag(sig**2))

    if store is not None:
        store_row = dict()
        store.append(store_row)

    def doSample(anneal=1.):
        results = OrderedDict()
        effdist = dist.Normal(eff_hat,
                    hat_data["var"])
        if anneal > 1:
            thetaSamplingDist = dist.Normal(eff_hat,
                        hat_data["var"] * anneal)
        else:
            thetaSamplingDist = effdist
        effects = pyro.sample('effects',
                        thetaSamplingDist.to_event(1))
        lpdf = effdist.log_prob(effects)
        results.update(lpdf=torch.sum(lpdf), effects=effects)


        if store is not None:
            store_row.update([(k,v.clone().detach()) for k,v in hat_data.items()])
        return results
    #
    doSample()
    return (eff_hat, torch.diag(sig), doSample)


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
    print("max",maxLogDens)
    return([(perp,pt,logDens-maxLogDens) for perp,pt,logDens in slices],maxLogDens)
    #return [(perp,pt,logDens) for perp,pt,logDens in slices] #not subtract maxLogDens because it's a nuisance later.

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

def adjustPoint(minDens,point,**kwargs): #Turns out the impact is epsilon, but whatevs.
    init = point.clone().detach().requires_grad_(True)


    hessCenter = pyro.condition(model,dict(effects=init))
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)(**kwargs)
    logPosterior = blockedTrace.log_prob_sum()

    loss = -((logPosterior - minDens) ** 2)[0]

    hess,grad = myhessian.hessian(loss, [init],
                return_grad=True,)#, allow_unused
    distance = -torch.mv(torch.inverse(hess),grad)
    result = init + distance * 0.9
    #complain("adjusted",result,point,logPosterior,minDens)
    return result

def adjustConfBounds(minDens,points,**kwargs):
    return [adjustPoint(minDens,point,**kwargs) for point in points]

def getConfBounds(alpha,slices,maxLogDens,**kwargs):
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
    adjustedMinDens = minDens + maxLogDens
    upperPart = adjustConfBounds(adjustedMinDens,
                  [pt for base,pt in northBorder] + #north
                  [2 * base - pt for base,pt in reversed(northBorder)], #west
                **kwargs)
    lowerPart = adjustConfBounds(adjustedMinDens,
                  [(2 * base - pt).index_select(0,swap) for base,pt in northBorder] + #south
                  [pt.index_select(0,swap) for base,pt in reversed(northBorder)], #east
                **kwargs)
    if slices[0][2] > minDens: #connected
        fullBorders = [upperPart + lowerPart]
    else: #separate
        fullBorders = [upperPart, lowerPart]
    return (fullBorders, minDens)

def getTruePosteriors(alphas,cutoff,delta,**kwargs):
    slices,maxLogDens = getPosteriorSlices(cutoff,delta,**kwargs)
    #print("slices",len(slices))
    curves = []
    for alpha in alphas:
        fullBorders,minDens = getConfBounds(alpha,slices,maxLogDens,**kwargs)
        curves.extend(fullBorders)
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


def getLaplaceParams(names = False):
    store = pyro.get_param_store()
    return getParams(store,("eff_hat"))

def getMeanfieldParams(names = False):
    store = pyro.get_param_store()
    return getParams(store,["auto_loc"],3)

def getParams(store,items,maxSize=1e3):
    result = []
    for item in items:
        try:
            vs = store[item].view(-1)[:maxSize]
            if names:
                for n,v in enumerate(vs):
                    result.append(item + str(n))
            else:
                for n,v in enumerate(vs):
                    result.append(float(v))
        except:
            pass

    return result


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
            filebase = None):

    stores = []
    if filebase is None:
        file = FakeSink()
    else:
        #
        filename = filebase + ".fitting.csv"
        file = open(filename,"a")             #leaking file handles. Who cares.
    writer = csv.writer(file)
    needs_header = True
    fitted_guides = OrderedDict()
    for guidename in guidenames:
        print(guidename,"vals",vals)
        guide = guides[guidename]


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
        base_names = ["guidename", "runtime",] + list(vals.keys())
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
            if filebase is None or os.stat(filename).st_size == 0: #header row
                if needs_header:
                    needs_header = False
                    try:
                        writer.writerow(base_names + ["i", "time", "loss"] + getLaplaceParams(True))
                    except:
                        writer.writerow(base_names + ["i", "time", "loss"] + getMeanfieldParams(True))

            if i % 10 == 0:
                try:
                    writer.writerow(base_line + [i, time.time(), loss] + getLaplaceParams())
                except:
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

        print("mean_loss:",mean_losses[-1] + .005)

        fitted_guides[guidename] = guide()
        #writer2.writerow(base_names )


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

    KLdivs(fitted_guides, vals, filebase)
    return(fitted_guides, vals, filebase)
    ##

def model_lps(sample,vals):
    hessCenter = pyro.condition(model,{"effects":sample})
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)(**vals)
    logPosterior = blockedTrace.log_prob_sum()
    return logPosterior


def KLdivs(guides, vals, filebase):
    print("KLdivs")
    filename = filebase + ".fitted.csv"
    with open(filename,"a") as file:
        writer = csv.writer(file)
        if os.stat(filename).st_size == 0: #header row
            writer.writerow( list(vals.keys()) + ["modelKL"] + list(guides.keys())) #header row

        def conditioned_model(model, *args, **kwargs):
            return poutine.condition(model, data={})(*args, **kwargs)

        nuts_kernel = NUTS(conditioned_model, jit_compile=False,)
        mcmc = MCMC(nuts_kernel,
                    num_samples=1000,
                    warmup_steps=50,
                    num_chains=1)
        mcmc.run(model, **vals)
        mcmc.summary(prob=0.5)

        guide_dists = [dist.MultivariateNormal(eff_hat,sig)
                                    for guidename, (eff_hat,sig,sampler) in guides.items()]

        print("KLdivs", filename, mcmc._samples["effects"].size())
        for sample in mcmc._samples["effects"]:
            logPosterior = model_lps(sample,vals)
            #
            writer.writerow( [float(x) for x in list(vals.values()) + [logPosterior] +
                    [guide_dist.log_prob(sample) for guide_dist in guide_dists]])
            writer.writerow( [float(x) for x in list(vals.values()) + [logPosterior] +
                    [guide_dist.log_prob(torch.flip(sample,[0])) for guide_dist in guide_dists]])
