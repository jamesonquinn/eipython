#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

from collections import OrderedDict
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

#Leaving this in broken state because I tested it out outside a model.
def model(G = 3, T = 4, N=ts([10,20,30,40]), O=3): #groups, subgroups, groupsize by trial, options
    print("model start")
    sdhyper = pyro.sample('sdhyper', dist.Gumbel(0.,1.))

    #gmeans = pyro.sample('gmeans', dist.StudentT(ts(7.).expand(G,G,O),0.,torch.exp(sdhyper)).to_event(3))
    gmeans = pyro.sample('gmeans', dist.StudentT(ts(7.).expand(G,O),0.,torch.exp(sdhyper)).to_event(2))
    print(gmeans)
    xs = []
    with pyro.plate('groups', G) as g:
        for t in range(T):
            print(g, T)
            xs.append(pyro.sample(f'x_{t}',
                    dist.Multinomial(int(N[t]),logits=gmeans).to_event(1)))
        print(f"model x_{t}:",xs[-1])
    penalty = pyro.sample('penalty', dist.Normal(0.,1.))
    print("penalty ",penalty)

    print("model end")

BASE_PSI =.01

def guide(G = 3, T = 4, N=ts([10,20,30,40]), O=3):
    print("guide start")


    hat_data = OrderedDict()
    ghat_data = OrderedDict()

    sdhyperhat = pyro.param('sdhyperhat', ts(0.))
    hat_data.update(sdhyper=sdhyperhat)
    gmeanshat = pyro.param('gmeanshat', torch.zeros(G,O))
    hat_data.update(gmeans=gmeanshat)
    lamhats = []
    #with pyro.plate('groups', G) as g:
    for t in range(T):
        lamhat = pyro.param(f'pi_hat_{t}',
                     torch.zeros(G,O))
        lamhats.append(lamhat)
        pihat = torch.exp(lamhat)
        ghat_data.update({
                    f'x_{t}':
                        N[t] * pihat / torch.sum(pihat,1).unsqueeze(1)
                    })
    print(f'x_{t}:',lamhats[t])

    combined_data = OrderedDict()
    combined_data.update(hat_data)
    combined_data.update(ghat_data)
    thetaMean = torch.cat([thetaPart.view(-1) for thetaPart in combined_data.values()],0)
    tlen = len(thetaMean)

    print("combined_data:")

    print(combined_data)

    #Get hessian
    hessCenter = pyro.condition(model,combined_data)
    blockedTrace = poutine.block(poutine.trace(hessCenter).get_trace)() #*args,**kwargs)
    logPosterior = blockedTrace.log_prob_sum()
    Info = -hessian.hessian(logPosterior, hat_data.values())#, allow_unused=True)

    #declare global-level psi params
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
    #


    with pyro.plate('groups', G) as g:
        for pname, phat in ghat_data.items():
            elems = phat.nelement()
            #print(f"adding {pname} from theta ({elems}, {phat.size()})" )
            pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
            pyro.sample(pname, dist.Delta(pdat.view(phat.size())).to_event(len(list(phat.size()))))


    print("guide end")


def trainGuide():
    svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO())

    pyro.clear_param_store()
    losses = []
    for i in range(3001):
        loss = svi.step()
        losses.append(loss)
        if i % 10 == 0:
            print(f'epoch {i} loss = {loss}')

    ##

    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    ##

    for (key, val) in sorted(pyro.get_param_store().items()):
        print(f"{key}:\n{val}")
