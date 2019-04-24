#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro import poutine
import hessian
import numpy as np
ts = torch.tensor


torch.manual_seed(478301986) #Gingles

pyro.enable_validation(True)
pyro.set_rng_seed(0)


def model(G = 3, N = 4):
    sdhyper = pyro.sample('sdhyper', dist.Gumbel(0.,1.))

    gmeans = pyro.sample('gmeans', dist.StudentT(ts(7.).expand([G]),0.,torch.exp(sdhyper)).to_event(1))
    gs = []
    for g in pyro.plate('groups', G):
        gs.append(pyro.sample(f'x_{g}',
                dist.Gumbel(gmeans[g].expand([N]),torch.exp(gmeans[(g+1)%G])).to_event(1)))
        print(f"model x_{g}:",gs[-1])

BASE_PSI =.01

def infoToM(Info,psi):
    tlen = len(psi)
    M = []
    for i in range(tlen):
        lseterms = torch.stack([ts(0.),
                            -Info[i,i] + psi[i],
                            -abs(Info[i,i]) + psi[i] +
                                torch.sum(torch.stack([abs(Info[i,j])
                                    for j in range(tlen) if j != i]))])
        M.append( psi[i] * torch.logsumexp(lseterms / psi[i],0))
    return torch.diag(torch.stack(M))

def guide(G = 3, N = 4):


    hat_data = dict()

    sdhyperhat = pyro.param('sdhyperhat', ts(0.))
    hat_data.update(sdhyper=sdhyperhat)
    gmeanshat = pyro.param('gmeanshat', torch.zeros(G))
    hat_data.update(gmeans=gmeanshat)
    gs = []
    for g in range(G):
        gs.append(pyro.param(f'xhat_{g}', torch.zeros(N)))
        hat_data.update({f'x_{g}':gs[g]})
        print(f'x_{g}:',gs[g])

    thetaMean = torch.cat([thetaPart.view(-1) for thetaPart in hat_data.values()],0)
    tlen = len(thetaMean)

    #Get hessian
    hessCenter = pyro.condition(model,hat_data)
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)() #*args,**kwargs)
    logPosterior = blockedTrace.log_prob_sum()
    Info = -hessian.hessian(logPosterior, hat_data.values())#, allow_unused=True)

    #declare global-level psi params
    globalpsi = pyro.param('globalpsi',torch.ones(tlen)*BASE_PSI,
                constraint=constraints.positive)
    M = infoToM(Info,globalpsi)
    adjusted = Info+M
    #print("matrix?",Info.size(),M.size(),[(float(Info[i,i]),float(M[i,i])) for i in range(tlen)])#,np.linalg.det(adjusted))
    theta = pyro.sample('theta',
                    dist.MultivariateNormal(thetaMean, precision_matrix=Info+M),
                    infer={'is_auxiliary': True})

    #decompose theta into specific values
    tmptheta = theta
    for pname, phat in hat_data.items():
        elems = phat.nelement()
        #print(f"adding {pname} from theta ({elems}, {phat.size()})" )
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))




def trainGuide():
    svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO())

    pyro.clear_param_store()
    losses = []
    for i in range(3001):
        loss = svi.step()
        losses.append(loss)
        if i % 100 == 0:
            print(f'epoch {i} loss = {loss}')

    ##

    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    ##

    for (key, val) in sorted(pyro.get_param_store().items()):
        print(f"{key}:\n{val}")
