#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
print('Yes, I will run.')

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from matplotlib import pyplot as plt

pyro.enable_validation(True)
pyro.set_rng_seed(0)

# Suppose we have aggregate level counts of two classes of voters and of votes in each of P precincts. Let's generate some fake data.

#...

# Now suppose precinct-level behavior follows a Beta distribution with class-dependent parameters, so that individual level behavior is Bernoulli distributed. We can write this as a generative model that we'll use both for data generation and inference.


def model(data=None, include_nuisance=False):
    if data is None:
        P, R, C = 10, 4, 3
        ns = torch.zeros(P,R)
        for p in range(P):
            for r in range(R):
                ns[p,r] = pyro.sample('precinctSizes', dist.NegativeBinomial(n + r + 1, .95))
    else:
        ns, vs = data
        assert len(ns)==len(vs)
        # Hyperparams.
        P = len(ns)
    sdc = 5
    sdrc = pyro.sample('sdrc', dist.Exponential(.2))
    sdprc = pyro.sample('sdprc', dist.Exponential(.2))

    with pyro.plate('candidates', C):
        ec = pyro.sample('ec', dist.Normal(0,sdc))
        with pyro.plate('rgroups', R):
            erc = pyro.sample('erc', dist.Normal(0,sdrc))
            if include_nuisance:
                with pyro.plate('precincts', P):
                    eprc = pyro.sample('eprc', dist.Normal(0,sdprc))

    logittotals = ec+erc
    if include_nuisance:
        logittotals += eprc

    y = zs(P,R,C)

    probs = zs(P,R,C)
    for p in pyro.plate('precincts', P):
        tmp = torch.exp(logittotals[p,r])
        cprobs = tmp/torch.sum(tmp)
        n = int(ns[p,r].item())
        samp= pyro.sample(f"y_{p}",
                    dist.Multinomial(n,cprobs)
                    .to_event(1))
        y[p,r] = samp



    if data is None:
        vs = torch.sum(y,1)

        return (ns,vs)


# Let's generate voting data. I'm conditioning concentration=1 for numerical stability.

data = model()
print(data)

# Let's now write a variational approximation.

init_narrow = 10  # Numerically stabilize initialization.

def guide(data, include_nuisance=False):

    ns, vs = data
    P = len(ns)
    R = len(ns[1])
    C = len(vs[1])
    #Start with hats.

    hat_data = dict()
    sdrchat = pyro.param('sdrchat',5)
    sdprchat = pyro.param('sdprchat',5)
    hat_data.update(sdrc=sdrchat,sdprc=sdprchat)
    with pyro.plate('candidates', C):
        echat = pyro.param('echat', 0)
        with pyro.plate('rgroups', R):
            erchat = pyro.param('erchat', 0)
            if include_nuisance:
                with pyro.plate('precincts', P):
                    eprchat = pyro.param('eprchat', 0)
    hat_data.update(ec=echat,erc=erchat,eprc=eprchat)


    with pyro.plate('precincts2', P):
        with pyro.plate('Wcandidates', C):
            with pyro.plate('Wrgroups', R):
                what = pyro.param('what', 0)

    for p in pyro.plate('precincts3', P):
        indep = get_indep(R, C, ns[p], vs[p])
        yhat = polytopize(R,C,indep,what[p])
        yy = pyro.param(f"y_{p}_hat",
                    yhat)
        hat_data.update({f"y_{p}":yy})
        if include_nuisance:
            pass #unimplemented — get MLE for gamma, yuck


    #Get hessians and sample params

    #Start with theta

    hessCenter = pyro.condition(model,hat_data)
    trace1 = poutine.trace(hessCenter)
    trace2 = trace1.get_trace(data, include_nuisance)
    loss = -trace2.log_prob_sum()
    thetaParts = [sdrchat, sdprchat, echat, erchat]
    H = hessian.hessian(loss, thetaParts)#, allow_unused=True)

    thetaMean = thetaParts.view(-1)
    tlen = len(thetaMean)
    theta = pyro.sample('theta',
                    dist.MultivariateNormal(thetaMean, precision_matrix=H),
                    infer={'is_auxiliary': True})


    for p in pyro.plate('precincts3', P):
        wthetaparts = thetaParts + [what[p]]
        HW = hessian.hessian(loss, wthetaParts)
        Sig = torch.inverse(H) #This is not efficient computationally — redundancy. But I don't want to hand-code the right thing yet.
        wmean = (what[p].view(-1) +
                Sig[tlen:, :tlen] * H * (theta - thetaMean))
        wSig = Sig[tlen:, tlen:] - Sig[tlen:, :tlen] * H * Sig[:tlen, tlen:]
        w = pyro.sample(f"w_{p}",
                        dist.MultivariateNormal(wmean, wSig),
                        infer={'is_auxiliary': True})
        indep = get_indep(R, C, ns[p], vs[p])
        y = polytopize(R,C,indep,w.view(R,C))
        yy = pyro.sample(f"y_{p}", dist.Delta(y))
        if include_nuisance:
            pass #unimplemented — gamma




def fritzguide(populations, data):

  # Posterior over global behavior patterns.

    alpha = pyro.param('alpha', init_narrow * torch.ones(2, 2))
    beta = pyro.param('beta', init_narrow * torch.ones(2, 2))
    concentration = pyro.sample('concentration', dist.Gamma(alpha,
                                beta).to_event(2))

  # Precint-level choices.

    with pyro.plate('precincts', P):

    # To sample from a subspace, we use an auxiliary sample site of lower
    # dimension, then inject it into the larger space via a Delta.

        t1 = pyro.param('t1', init_narrow * torch.ones(P))
        t0 = pyro.param('t2', init_narrow * torch.ones(P))
        t = pyro.sample('t', dist.Beta(t1, t0),
                        infer={'is_auxiliary': True}).unsqueeze(-1)

    # Let t in [0,1] parametrize the range of possible behaviors
    #   data = behavior[:,0] * x + behavior[:,1] * y
    # where x,y are local population portions so that x+y=1.

        x = populations[:, 0] / populations.sum(-1)
        y = populations[:, 1] / populations.sum(-1)
        behavior_min = torch.stack([torch.min(data, x),
                                   torch.max(torch.tensor(0.), data
                                   - x)], dim=-1)
        behavior_max = torch.stack([torch.max(torch.tensor(0.), 1
                                   - data - y), torch.min(1 - data,
                                   y)], dim=-1)
        behavior = behavior_min + (behavior_max - behavior_min) * t
        pyro.sample('behavior', dist.Delta(behavior, event_dim=1))


# Now let's train the guide.

svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO())

pyro.clear_param_store()
losses = []
for i in range(3001):
    loss = svi.step(populations, data)
    losses.append(loss)
    if i % 100 == 0:
        print('epoch {} loss = {}'.format(i, loss))

##

plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('loss')

##

for (key, val) in sorted(pyro.get_param_store().items()):
    print(f"{key}:\n{val}")
