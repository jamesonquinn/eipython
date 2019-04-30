#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
print('Yes, I will run.')

from importlib import reload
import go_or_nogo
import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from matplotlib import pyplot as plt
from cmult import CMult
from polytopize import get_indep, polytopize, depolytopize
from pyro import poutine
import hessian
import numpy as np
import cProfile as profile
ts = torch.tensor


torch.manual_seed(478301986) #Gingles

pyro.enable_validation(True)
pyro.set_rng_seed(0)

# Suppose we have aggregate level counts of two classes of voters and of votes in each of P precincts. Let's generate some fake data.

#...

# Now suppose precinct-level behavior follows a Beta distribution with class-dependent parameters, so that individual level behavior is Bernoulli distributed. We can write this as a generative model that we'll use both for data generation and inference.


def model(data=None, include_nuisance=False):
    if data is None:
        P, R, C = 30, 4, 3
        ns = torch.zeros(P,R)
        for p in range(P):
            for r in range(R):
                ns[p,r] = 20 * (p*r - 5*(r+1-R) + 6) + ((p-15)*(r-1))^2

                # pyro.sample('precinctSizes',
                #         dist.NegativeBinomial(p*r - 5*(r+1-R) + 6, .95))
    else:
        ns, vs = data
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
        sdc = .2
        sdrc = .4
        sdprc = .6

    ec = pyro.sample('ec', dist.Normal(torch.zeros(C),sdc).to_event(1))
    erc = pyro.sample('erc', dist.Normal(torch.zeros(R,C),sdrc).to_event(2))
    if include_nuisance:
        eprc = pyro.sample('eprc', dist.Normal(torch.zeros(P,R,C),sdprc).to_event(3))

    if data is None:
        erc = torch.zeros([R,C])
        erc[0] = ts(range(C))
        erc[1,0] = ts(2.)

    # with pyro.plate('candidatesm', C):
    #     ec = pyro.sample('ec', dist.Normal(0,sdc))
    #     with pyro.plate('rgroupsm', R):
    #         erc = pyro.sample('erc', dist.Normal(0,sdrc))
    #         if include_nuisance:
    #             with pyro.plate('precinctsm', P):
    #                 eprc = pyro.sample('eprc', dist.Normal(0,sdprc))

    logittotals = ec+erc
    #print("sizes ",P,R,C,ec.size(),erc.size(),logittotals.size())
    if include_nuisance:
        logittotals += eprc
    else:
        logittotals = torch.cat([logittotals.unsqueeze(0) for p in range(P)],0)
        #print(logittotals.size())

    y = torch.zeros(P,R,C)

    for p in range(P):#pyro.plate('precinctsm2', P):
        for r in range(R):#pyro.plate('rgroupsm2', R):
            tmp = torch.exp(logittotals[p,r])
            cprobs = tmp/torch.sum(tmp,0)
            #print("cprobs ",cprobs)
            y[p,r]= pyro.sample(f"y_{p}_{r}",
                        CMult(ns[p,r],cprobs))



    if data is None:
        #
        print(f"ec:{ec}")
        print(f"erc:{erc}")
        print(f"y[0]:{y[0]}")
        vs = torch.sum(y,1)

        return (ns,vs)



# Let's now write a variational approximation.

init_narrow = 10  # Numerically stabilize initialization.
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

BUNCHFAC = 9999
#P=10, BUNCHFAC = 9999: 61/31
#P=10, BUNCHFAC = 1: 110/31
#P=30, BUNCHFAC = 1: 785/31; 2490..1799..1213
#P=30, BUNCHFAC = 1: 675/31; 2490..1799..1213
#P=30, BUNCHFAC = 2: 339/31
#P=30, BUNCHFAC = 3: 96/11
#P=30, BUNCHFAC = 9999: 42/11
#P=30, BUNCHFAC = 9999: 189/51 2346..1708..1175..864..746

def recenter_rc(rc):
    return (rc - .95 * torch.mean(rc,0))

def guide(data, include_nuisance=False):

    ns, vs = data
    P = len(ns)
    R = len(ns[1])
    C = len(vs[1])

    #declare precinct-level psi params
    precinctpsi = pyro.param('precinctpsi',BASE_PSI * torch.ones((R-1)*(C-1)),
                constraint=constraints.positive)


    #Start with hats.

    hat_data = dict()
    transformation = dict()
    logsdrchat = pyro.param('logsdrchat',ts(2.))
    hat_data.update(sdrc=logsdrchat)
    transformation.update(sdrc=torch.exp)
    if include_nuisance:
        logsdprchat = pyro.param('logsdprchat',ts(2.))
        hat_data.update(sdprc=logsdprchat)
        transformation.update(sdprc=torch.exp)

    echat = pyro.param('echat', torch.zeros(C))
    erchat = pyro.param('erchat', torch.zeros(R,C))
    hat_data.update(ec=echat,erc=erchat)
    transformation.update(erc=recenter_rc)
    # with pyro.plate('candidatesg', C):
    #     echat = pyro.param('echat', ts(0.))
    #     with pyro.plate('rgroupsg', R):
    #         erchat = pyro.param('erchat', ts(0.))
    #         if include_nuisance:
    #             with pyro.plate('precinctsg', P):
    #                 eprchat = pyro.param('eprchat', ts(0.))
    #             hat_data.update(eprc=eprchat)

    global_logit_totals =

    what = []

    for p in range(P):#pyro.plate('precinctsg2', P):
        pwhat = pyro.param(f'what_{p}', torch.zeros(R-1,C-1))
        what.append(pwhat)
        indep = get_indep(R, C, ns[p], vs[p])
        yhat = polytopize(R, C, pwhat, indep)
        for r in range(R):
            #print(f"yy size:{yy.size()},{R},{C}")
            hat_data.update({f"y_{p}_{r}":yhat[r]}) #no, don't; do it separately.
            if include_nuisance:
                pass #unimplemented — get MLE for gamma, yuck


    #Get hessians and sample params

    #Start with theta

    transformed_hat_data = dict()
    for k,v in hat_data.items():
        if k in transformation:
            transformed_hat_data[k] = transformation[k](v)
        else:
            transformed_hat_data[k] = v
    hess_center = pyro.condition(model,transformed_hat_data)
    mytrace = poutine.block(poutine.trace(hess_center).get_trace)(data, include_nuisance)
    log_posterior = mytrace.log_prob_sum()
    theta_part_names = ["sdrc", "sdprc", "ec", "erc", "eprc"]
    theta_parts = []
    theta_hat_data = dict()
    for part_name in theta_part_names:
        if part_name in hat_data:
            theta_parts.append(hat_data[part_name]) #fails if missing (ie, eprc)
            theta_hat_data[part_name] = hat_data[part_name]

    Info = -hessian.hessian(log_posterior, theta_parts)#, allow_unused=True)

    theta_mean = torch.cat([tpart.view(-1) for tpart in theta_parts],0)
    tlen = len(theta_mean)

    #declare global-level psi params
    globalpsi = pyro.param('globalpsi',torch.ones(tlen)*BASE_PSI,
                constraint=constraints.positive)
    M = infoToM(Info,globalpsi)
    adjusted = Info+M
    #print("matrix?",Info.size(),M.size(),[(float(Info[i,i]),float(M[i,i])) for i in range(tlen)])#,np.linalg.det(adjusted))
    theta = pyro.sample('theta',
                    dist.MultivariateNormal(theta_mean, precision_matrix=Info+M),
                    infer={'is_auxiliary': True})

    #decompose theta into specific values
    tmptheta = theta
    for pname, phat in theta_hat_data.items():
        elems = phat.nelement()
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        #print(f"adding {pname} from theta ({elems}, {phat.size()}, {tmptheta.size()}, {pdat})" )
        if pname in transformation:
            pyro.sample(pname, dist.Delta(transformation[pname](pdat.view(phat.size())))
                                .to_event(len(list(phat.size()))))
        else:
            pyro.sample(pname, dist.Delta(pdat.view(phat.size()))
                                .to_event(len(list(phat.size()))))
    assert list(tmptheta.size())[0] == 0


    combinedpsi = torch.cat([globalpsi, precinctpsi],0)

    #
    B = tlen * BUNCHFAC // (R-1) // (C-1) #bunch size; should do enough precincts to approximately double total matrix size
    nbunches = ((P-1) // B) + 1
    big_HWs = []
    for b in range(nbunches):
        wtheta = theta_parts + what[(b*B):((b+1)*B)]
        big_HWs.append(hessian.hessian(log_posterior, wtheta))
    #print(big_HW.size())
    precinct_indices = []
    for pp in range(B): #possible remainders
         precinct_indices.append(ts(range(tlen + pp*(R-1)*(C-1), tlen + (pp+1)*(R-1)*(C-1))))
    global_indices = ts(range(tlen))
    for p in range(P):#pyro.plate('precincts3', P):
        pp = p % B #remainder
        #print(f"Hesshess:{hessian.hessian(log_posterior,theta_parts + [what[p]])}")

        full_indices = torch.cat([global_indices,precinct_indices[pp]],0)
        HW = big_HWs[p // B].index_select(0,full_indices).index_select(1,full_indices)
        #print(f"size:{HW.size()},{tlen},{len(precinct_indices)},{len(combinedpsi)}")
        M = infoToM(HW,combinedpsi)
        precision = HW + M
        Sig = torch.inverse(precision) #This is not efficient computationally — redundancy.
                        #But I don't want to hand-code the right thing yet.
        #print(f"substep:{tlen},{Info.size()},{Sig.size()}")
        substep = torch.mm(Sig[tlen:, :tlen], adjusted)

        wmean = (what[p].view(-1) +
                torch.mv(substep, (theta - theta_mean)))
        wSig = Sig[tlen:, tlen:] - torch.mm(substep, Sig[:tlen, tlen:])
        wSig = wSig + infoToM(wSig, combinedpsi[tlen:])
        try:
            w = pyro.sample(f"w_{p}",
                            dist.MultivariateNormal(wmean, wSig),
                            infer={'is_auxiliary': True})
        except:
            print(wSig)
            print(f"det:{np.linalg.det(wSig.data.numpy())}")
            raise
        indep = get_indep(R, C, ns[p], vs[p])
        y = polytopize(R,C,w.view(R-1,C-1),indep)
        for r in range(R):
            yy = pyro.sample(f"y_{p}_{r}", dist.Delta(y[r]).to_event(1))
        if include_nuisance:
            pass #unimplemented — gamma


nsteps = 5001

def trainGuide():

    # Let's generate voting data. I'm conditioning concentration=1 for numerical stability.

    data = model()
    print(data)


    # Now let's train the guide.
    svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO())

    pyro.clear_param_store()
    losses = []
    for i in range(nsteps):
        loss = svi.step(data)
        losses.append(loss)
        if i % 10 == 0:
            print(f'epoch {i} loss = {loss};'+
                f' logsdrchat = {pyro.get_param_store()["logsdrchat"]};' +
                f' erchat[0] = {pyro.get_param_store()["erchat"][0]}')
            reload(go_or_nogo)
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
    print("yhat[0]:",
        polytopize(4,3,pyroStore["what_0"],
                   get_indep(4,3,ns[0],vs[0])))

    return(svi,losses,data)
