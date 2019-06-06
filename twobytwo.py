#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function


from importlib import reload
import contextlib
from itertools import chain
import cProfile as profile
import inspect
from collections import OrderedDict


from matplotlib import pyplot as plt
import numpy as np
import csv
import time
import math
import pandas as pd

import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro import poutine


import myhessian
from rank1torch import optimize_Q
import go_or_nogo
from cmult import CMult
import polytopize
reload(polytopize)
from polytopize import get_indep, polytopize, depolytopize, to_subspace, process_data

ts = torch.tensor

pyro.enable_validation(True)
pyro.set_rng_seed(0)

VERSION = "0.1"

BASE_PSI = .01

EULER_CONSTANT = 0.5772156649015328606065120900824024310421
GUMBEL_SD = math.pi/math.sqrt(6.)


MINIMAL_DEBUG = False
if MINIMAL_DEBUG:
    pyrosample = lambda x,y,infer=None : y.sample()
else:
    pyrosample = pyro.sample

def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno


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



def model(data=None, scale=1., include_nuisance=False, do_print=False,
            fixedParams = None): #groups, subgroups, groupsize by trial, options

    if data is None:
        P, R, C = 30, 2, 2
        ns = torch.zeros(P,R)
        for p in range(P):
            for r in range(R):
                ns[p,r] = 600*(1-r) + ((p+1)**2)*r*10
    else:
        ns, vs, indeps, tots = data
        assert len(ns)==len(vs)
        # Hyperparams.
        P = len(ns)
        R = len(ns[0])
        C = len(vs[0])


    sdc = 5
    sdrc = pyro.sample('sdrc', dist.Exponential(.2))
    if include_nuisance:
        sdprc = pyro.sample('sdprc', dist.Exponential(.2))

    if data is None:
        if fixedParams is not None:
            sdc = fixedParams['sdc']
            sdrc = fixedParams['sdrc']
            sdprc = fixedParams['sdprc']
        else:
            sdc = scrc = sdprc = ts(1.)

    #This is NOT used for data, but instead as a way to sneak a "prior" into the guide to improve identification.
    param_residual=pyro.sample('param_residual', dist.Normal(0.,1.))


    ec = pyro.sample('ec', dist.Normal(torch.zeros(C),sdc).to_event(1))
    erc = pyro.sample('erc', dist.Normal(torch.zeros(R,C),sdrc).to_event(2))


    prepare_ps = range(P)
    all_ps_plate = pyro.plate('all_ps',P)
    @contextlib.contextmanager
    def all_ps():
        with all_ps_plate as p:#, poutine.scale(scale=scale) as pscale:
            yield p
    if include_nuisance:
        with all_ps() as p_tensor:
            eprc = (
                pyro.sample(f'eprc', dist.Normal(torch.zeros(P,R,C),sdprc).to_event(3))
                ) #eprc.size() == [P,R,C] because plate dimension happens on left


    if data is None:
        if fixedParams is not None and "erc" in fixedParams:
            erc = fixedParams['erc']
            ec = fixedParams['ec']
            try:
                eprc = fixedParams['eprc']
            except:
                pass
        else:
            erc = torch.zeros(R,C)
            erc[0] = ts(range(C))
            erc[1,0] = ts(2.)
            ec = torch.zeros(C)
            ec[1] = .5
            eprc= torch.zeros(P,R,C)


    logits = ec+erc
    #print("sizes1 ",P,R,C,eprc.size(),ec.size(),erc.size(),logittotals.size())
    if include_nuisance:
        logits = logits + eprc # += doesn't work here because mumble in-place mumble shape
    else:
        logits = logits.expand(P,R,C)


    with all_ps() as p_tensor:#pyro.plate('precinctsm2', P):
        #with poutine.scale(scale=scale): #TODO: insert!
        if data is None:
            y = torch.zeros(P,R,C)
            for p in p_tensor:
                for r in range(R):
                    tmp = dist.Multinomial(int(ns[p,r]),logits=logits[p,r]).sample()
                    #print(f"y_{p}_{r}: {tmp} {y[p,r]}")
                    y[p,r] = tmp
        else:
            y = pyro.sample(f"y",
                        CMult(1000,logits=logits).to_event(1))
                        #dim P, R, C from plate, to_event, CMult
                        #note that n is totally fake — sums are what matter.
                        #TODO: fix CMult so this fakery isn't necessary.


    if data is None:
        #
        print(f"ec:{ec}")
        print(f"erc:{erc}")
        print(f"y[0]:{y[0]}")
        vs = torch.sum(y,1)

        return (ns,vs)






















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

def amortized_laplace(data=None, scale=1., include_nuisance=False, do_print=False,
            fixedParams = None):


    ns, vs, indeps, tots = data
    P = len(ns)
    R = len(ns[1])
    C = len(vs[1])
    assert R == 2
    assert C == 2


    prepare_ps = range(P) #for dealing with hatted quantities (no pyro.sample)
    all_ps_plate = pyro.plate('all_ps',P)
    @contextlib.contextmanager
    def all_ps(): #for dealing with unhatted quantities (include pyro.sample)
        with all_ps_plate as p, poutine.scale(scale=scale) as pscale:
            yield p


    hat_data = OrderedDict()
    #this will hold values to condition the model and get the Hessian

    phat_data = OrderedDict()
    #as above, but not part of theta

    transformation = OrderedDict()
    logsdrchat = pyro.param('logsdrchat',ts(2.))
    hat_data.update(sdrc=logsdrchat)
    transformation.update(sdrc=torch.exp)
    if include_nuisance:
        logsdprchat = pyro.param('logsdprchat',ts(2.))
        hat_data.update(sdprc=logsdprchat)
        transformation.update(sdprc=torch.exp)
        eprchat_startingpoint = torch.zeros(P,R,C,requires_grad =True) #not a pyro param...
        #eprchat_startingpoint[p].requires_grad_(True) #...so we have to do this manually
        phat_data.update(eprc=eprchat_startingpoint)


    echat = pyro.param('echat', torch.zeros(C))
    erchat = pyro.param('erchat', torch.zeros(R,C))
    hat_data.update(ec=echat,erc=erchat)
    #transformation.update(erc=recenter_rc) #not do this.


    #not fully identifiable, so "impose a prior"
    recentering_amount = torch.abs(torch.sum(echat)) + sum(torch.abs(torch.sum(erchat[r,:])) for r in range(R))
    pyro.sample('param_residual', dist.Delta(recentering_amount * .05))

    #get ready to amortize: detach
    detached_hat_data = OrderedDict()
    for paramname, param in hat_data.items():
        detached_hat_data[paramname] = param.detach().requires_grad_()

    #Amortize hats
    yhat = []
    what = []
    nuhat = []


    logittotals = detached_hat_data["ec"] + detached_hat_data["erc"]
    #print("sizes1 ",P,R,C,eprc.size(),ec.size(),erc.size(),logittotals.size())
    if include_nuisance:
        logittotals = logittotals + eprchat_startingpoint # += doesn't work here because mumble in-place mumble shape
        #note that startingpoint is currently zero, so this is effectively just unsqueeze
        #but still healthy to do it like this in case that changes later.
    else:
        logittotals = logittotals.expand(P,R,C)
    pi_raw = torch.exp(logittotals)
    pi = pi_raw / torch.sum(pi_raw,-1).unsqueeze(-1)
    #print("pi:",pi)


    #print("guide:pre-p")
    for p in prepare_ps:#pyro.plate('precinctsg2', P):
        #precalculation - logits to pi


        #get ŷ^(0)
        Q, iters = optimize_Q(R,C,pi[p],vs[p],ns[p],tolerance=.01,maxiters=3)
        #print(f"optimize_Q {p}:{iters}")
        yhat.append(Q*tots[p])

        #depolytopize
        what.append(depolytopize(R,C,yhat[p],indeps[p]))

        #get ν̂^(0)
        if include_nuisance:
            pi_precision = tots[p] / pi[p] / (torch.ones_like(pi) - pi)
            Q_precision = torch.exp(-logsdprchat) * R / (R-1) #TODO: check that correction for R
            pweighted_Q = pi[p] + (Q-pi[p]) * Q_precision / (Q_precision + pi_precision)
            nuhat.append(torch.log(pweighted_Q/pi[p]))


    if include_nuisance:
        phat_data.update(y=torch.cat(yhat,0))

    transformed_hat_data = OrderedDict()
    for k,v in chain(hat_data.items(),phat_data.items()):
        if k in transformation:
            transformed_hat_data[k] = transformation[k](v)
        else:
            transformed_hat_data[k] = v


    real_hessian = not MINIMAL_DEBUG
    if real_hessian:
        #
        #print("line ",lineno())
        hess_center = pyro.condition(model,transformed_hat_data)
        #print("line ",lineno())
        mytrace = poutine.block(poutine.trace(hess_center).get_trace)(data, scale, include_nuisance)
        #print("line ",lineno())
        log_posterior = mytrace.log_prob_sum()
        #print("line ",lineno())

    theta_parts = list(hat_data.values())

    tlen = sum(theta_part.numel() for theta_part in theta_parts)

    if include_nuisance:
        blocksize = 2 #tensors, not elements=(R-1)*(C-1) + R*C
        precelems = (R-1)*(C-1) + R*C
        for p in prepare_ps:
            theta_parts.extend([what[p], hat_data[f"eprc_{p}"]])
    else:
        blocksize = 1 #tensors, not elements=(R-1)*(C-1)
        precelems = (R-1)*(C-1)
        for p in prepare_ps:
            theta_parts.extend([what[p]])
    full_len = sum(theta_part.numel() for theta_part in theta_parts)


    if real_hessian:
        neg_big_hessian, big_grad = myhessian.arrowhead_hessian(log_posterior, theta_parts,
                    len(hat_data), #tensors, not elements=tlen
                    blocksize,
                    return_grad=True,
                    allow_unused=True)
    else:
        neg_big_hessian, big_grad = -torch.eye(full_len), torch.zeros(full_len)

    big_hessian = -neg_big_hessian + infoToM(-neg_big_hessian)#TODO: in-place
    theta_cov = get_unconditional_cov(big_hessian,tlen)


    all_means = torch.cat([tpart.view(-1) for tpart in theta_parts],0)
    theta_mean = all_means[:tlen]

    #sample top-level parameters
    theta = pyro.sample('theta',
                    dist.MultivariateNormal(theta_mean,
                                theta_cov),
                    infer={'is_auxiliary': True})

    #decompose theta into specific values
    tmptheta = theta
    for pname in hat_data.keys():
        phat = hat_data[pname]
        elems = phat.nelement()
        #print(f"pname, phat: {pname}, {phat}")
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        #print(f"adding {pname} from theta ({elems}, {phat.size()})" )

        if pname in transformation:
            pyro.sample(pname, dist.Delta(transformation[pname](pdat.view(phat.size())))
                                .to_event(len(list(phat.size())))) #TODO: reshape after transformation, not just before???
        else:
            pyro.sample(pname, dist.Delta(pdat.view(phat.size()))
                                .to_event(len(list(phat.size()))))
    assert list(tmptheta.size())[0] == 0

    #sample unit-level parameters, conditional on top-level ones
    global_indices = ts(range(tlen))
    with all_ps() as p_tensor:
        for p in p_tensor:
        #
        #
            precinct_indices = ts(range(tlen + p*precelems, tlen + (p+1)*precelems))


            full_indices = torch.cat([global_indices, precinct_indices],0)

            full_precision = big_hessian.index_select(0,full_indices).index_select(1,full_indices) #TODO: do in-place?
            full_mean = all_means.index_select(0,full_indices) #TODO: do in-place!
            new_mean, new_precision = conditional_normal(full_mean, full_precision, tlen, theta)


            pparam_list = []
            try:
                pparam_list.append( pyro.sample(f"p_{p}",
                                dist.MultivariateNormal(new_mean, precision_matrix=new_precision),
                                infer={'is_auxiliary': True}))
            except:
                print("error conditionally drawing precinct")
                print(new_mean,new_precision)
                print(f"det:{np.linalg.det(new_precision.data.numpy())}")
                print(full_mean)
                print(all_means)
                raise

    ys = torch.cat([polytopize(R,C,y[:1].view(R-1,C-1),indep).view(1,R,C)
                        for indep,y in zip(indeps,pparam_list)],0)
    pyro.sample("y", dist.Delta(ys))
    if include_nuisance:
        eprc = torch.cat([y[1:].view(1,R,C) for y in pparam_list],0)
        pyro.sample("eprc", dist.Delta(eprc))
    #
    #print("end guide.",theta[:3],mode_hat,nscale_hat,gscale_hat,Info[:5,:5],Info[-3:,-3:])
    #
    #print(".....1....",true_g_hat,theta[-6:])
    #print(".....2....",theta[-9:-6])
    grads_unfixed = True
    def fix_grads():
        if grads_unfixed:
            grads_unfixed = False
            for paramname, param in hat_data.items():
                param.grad = param.grad + detached_hat_data[paramname].grad


    for paramname, param in hat_data.items():
        param.fix_grad = fix_grads

    #


class FakeSink(object):
    def write(self, *args):
        pass
    def writelines(self, *args):
        pass
    def close(self, *args):
        pass

def trainGuide(
            nparticles = 1,
            filename = None):

    guide = amortized_laplace
    data = model()

    processed_data = process_data(data)
    if filename is None:
        file = FakeSink()
    else:
        file = open(filename,"a")
    writer = csv.writer(file)

    #guide = guide2
    svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO(nparticles))
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'betas': (0.99,0.9999)}), Trace_ELBO(nparticles)) #.72
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'betas': (0.8,0.9)}), Trace_ELBO(nparticles)) #?
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'clip_norm': 5.0}), Trace_ELBO(nparticles)) #.66
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'weight_decay': ...}), Trace_ELBO(nparticles))
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'eps': 1e-5}), Trace_ELBO(nparticles))
    #svi = SVI(model, guide, ClippedAdam({'lr': 0.005, 'eps': 1e-10}), Trace_ELBO(nparticles))
    #svi = SVI(model, guide, AdagradRMSProp({}), Trace_ELBO(nparticles))

    pyro.clear_param_store()
    losses = []
    mean_losses = [] #(moving average)
    runtime = time.time()
    base_line = [VERSION, runtime,]
                    # [trueparams[item] for item in ("modal_effect",
                    #                 "norm_scale","gum_scale")]]
    for i in range(3001):
        loss = svi.step(processed_data)
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
