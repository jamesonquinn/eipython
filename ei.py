#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
print('Yes, I will run.')

from importlib import reload
import contextlib
from itertools import chain
import go_or_nogo
import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from matplotlib import pyplot as plt
from cmult import CMult
import polytopize
reload(polytopize)
from polytopize import get_indep, polytopize, depolytopize, to_subspace
from pyro import poutine
import myhessian
import numpy as np
import cProfile as profile
ts = torch.tensor


torch.manual_seed(478301986) #Gingles

pyro.enable_validation(True)
pyro.set_rng_seed(0)

# Suppose we have aggregate level counts of two classes of voters and of votes in each of P precincts. Let's generate some fake data.

#...

# Now suppose precinct-level behavior follows a Beta distribution with class-dependent parameters, so that individual level behavior is Bernoulli distributed. We can write this as a generative model that we'll use both for data generation and inference.

MINIMAL_DEBUG = False
if MINIMAL_DEBUG:
    pyrosample = lambda x,y,infer=None : y.sample()
else:
    pyrosample = pyro.sample

def model(data=None, scale=1., include_nuisance=False, do_print=False):
    #print("model:begin")
    if data is None:
        P, R, C = 30, 4, 3
        ns = torch.zeros(P,R)
        for p in range(P):
            for r in range(R):
                ns[p,r] = 20 * (p*r - 5*(r+1-R) + 6) + ((p-15)*(r-1))^2

                    # pyrosample('precinctSizes',
                    #         dist.NegativeBinomial(p*r - 5*(r+1-R) + 6, .95))
    else:
        ns, vs = data
        assert len(ns)==len(vs)
        # Hyperparams.
        P = len(ns)
        R = len(ns[0])
        C = len(vs[0])

    prepare_ps = range(P)
    all_ps_plate = pyro.plate('all_ps',P)
    @contextlib.contextmanager
    def all_ps():
        with all_ps_plate as p, poutine.scale(scale=scale) as pscale:
            yield p

    sdc = 5
    sdrc = pyrosample('sdrc', dist.Exponential(.2))
    if include_nuisance:
        sdprc = pyrosample('sdprc', dist.Exponential(.2))

    if data is None:
        sdc = .2
        sdrc = .4
        sdprc = .6
    #print(f"sdprc in model:{sdprc}")

    #This is NOT used for data, but instead as a way to sneak a "prior" into the guide to improve identification.
    param_residual=pyrosample('param_residual', dist.Normal(0.,1.))

    ec = pyrosample('ec', dist.Normal(torch.zeros(C),sdc).to_event(1))
    erc = pyrosample('erc', dist.Normal(torch.zeros(R,C),sdrc).to_event(2))
    if include_nuisance:
        with all_ps() as p_tensor:
            eprc = (
                pyrosample(f'eprc', dist.Normal(torch.zeros(R,C),sdprc).to_event(2))
                ) #eprc.size() == [P,R,C] because plate dimension happens on left
        eprc = torch.stack(eprc_list,0)
    else:
        eprc = ts(0.) #dummy for print statements. TODO:remove

    if data is None:
        erc = torch.zeros([R,C])
        erc[0] = ts(range(C))
        erc[1,0] = ts(2.)
        ec = torch.zeros(C)
        ec[1] = .5

    # with pyro.plate('candidatesm', C):
    #     ec = pyrosample('ec', dist.Normal(0,sdc))
    #     with pyro.plate('rgroupsm', R):
    #         erc = pyrosample('erc', dist.Normal(0,sdrc))
    #         if include_nuisance:
    #             with pyro.plate('precinctsm', P):
    #                 eprc = pyrosample('eprc', dist.Normal(0,sdprc))

    logittotals = ec+erc
    #print("sizes1 ",P,R,C,eprc.size(),ec.size(),erc.size(),logittotals.size())
    if include_nuisance:
        logittotals = logittotals + eprc # += doesn't work here because mumble in-place mumble shape
    else:
        logittotals = torch.cat([logittotals.unsqueeze(0) for p in range(P)],0)
        #print(logittotals.size())
    #print("sizes2 ",P,R,C,ec.size(),erc.size(),logittotals.size(),scale)

    if include_nuisance:
        logits = ec + erc + eprc
    else:
        logits = (ec + erc).expand(P,-1,-1)#P,R,C
    with all_ps() as p_tensor:#pyro.plate('precinctsm2', P):
        #with poutine.scale(scale=scale): #TODO: insert!
        if data is None:
            y = zeros(P,R,C)
            for p in p_tensor:
                for r in range(R):
                    y[p,r] = pyrosample(f"FAKE_y_{p}_{r}",
                                CMult(ns[p,r],logits=logits))
        else:
            y = pyrosample(f"y",
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

    print("model:end")



# Let's now write a variational approximation.

init_narrow = 10  # Numerically stabilize initialization.
BASE_PSI =.01

def infoToM(Info,psi):
    tlen = len(psi)
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

BUNCHFAC = 35
#P=10, BUNCHFAC = 9999: 61/31
#P=10, BUNCHFAC = 1: 110/31
#P=30, BUNCHFAC = 1: 785/31; 2490..1799..1213
#P=30, BUNCHFAC = 1: 675/31; 2490..1799..1213
#P=30, BUNCHFAC = 2: 339/31
#P=30, BUNCHFAC = 3: 96/11
#P=30, BUNCHFAC = 9999: 42/11
#P=30, BUNCHFAC = 9999: 189/51 2346..1708..1175..864..746

ADJUST_SCALE = .05
MAX_NEWTON_STEP = 1.2
RECENTER_PRIOR_STRENGTH = 2.

def recenter_rc(rc):
    rowcentered= (rc - torch.mean(rc,0))
    colcentered = rowcentered - torch.mean(rowcentered,0)
    return colcentered

def guide(data, scale, include_nuisance=False, do_print=False):
    #print("guide:begin")

    ns, vs = data
    P = len(ns)
    R = len(ns[1])
    C = len(vs[1])

    if include_nuisance:
        precinctpsi_len = (R-1)*(C-1) + R*C
    else:
        precinctpsi_len = (R-1)*(C-1)
    #declare precinct-level psi params
    precinctpsi = pyro.param('precinctpsi',BASE_PSI * torch.ones(precinctpsi_len),
                constraint=constraints.positive)

    prepare_ps = range(P) #for dealing with hatted quantities (no pyro.sample)
    all_ps_plate = pyro.plate('all_ps',P)
    @contextlib.contextmanager
    def all_ps(): #for dealing with unhatted quantities (include pyro.sample)
        with all_ps_plate as p, poutine.scale(scale=scale) as pscale:
            yield p

    #Start with hats.

    hat_data = dict()
    phat_data = dict()
    transformation = dict()
    logsdrchat = pyro.param('logsdrchat',ts(2.))
    hat_data.update(sdrc=logsdrchat)
    transformation.update(sdrc=torch.exp)
    if include_nuisance:
        logsdprchat = pyro.param('logsdprchat',ts(2.))
        hat_data.update(sdprc=logsdprchat)
        transformation.update(sdprc=torch.exp)
        eprchat_startingpoint = torch.zeros(P,R,C,requires_grad =True) #not a pyro param...
        #eprchat_startingpoint[p].requires_grad_(True) #...so we have to do this manually
        phat_data.update({f"eprc":eprchat_startingpoint})

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

    #Including recenter_rc makes erc not identifiable, so "impose a prior"
    recentering_amount = (RECENTER_PRIOR_STRENGTH *
            torch.mean((erchat - recenter_rc(erchat))**2)/torch.exp(logsdrchat))
    pyrosample('param_residual', dist.Delta(recentering_amount))

    global_logit_totals = echat + erchat

    global_probs = []
    for r in range(R):#pyro.plate('rgroupsm2', R):
        tmp = torch.exp(global_logit_totals[r])
        global_probs.append(tmp/torch.sum(tmp,0))

    gprobs_tensor = torch.stack(global_probs,0)
    #print(f"gprobs_tensor,{gprobs_tensor.size()},{torch.unsqueeze(ns[p],1).size()}")


    what = []
    what_adjustor = pyro.param('what_adjustor',torch.zeros(P,R-1,C-1))


    for p in prepare_ps:#pyro.plate('precinctsg2', P):
        raw_w = gprobs_tensor * torch.unsqueeze(ns[p],1)
        pwhat = to_subspace(raw_w,R,C,ns[p],vs[p])
        #print("cprobs ",cprobs)
        what.append(pwhat[:R-1,:C-1] + what_adjustor[p] * torch.sum(ns[p]) * ADJUST_SCALE)
        indep = get_indep(R, C, ns[p], vs[p])
        yhat = polytopize(R, C, what[p], indep) #was: pwhat, indep, do_aug=False)
                #but that confuses hessian, so remove and re-add? inefficient but whatevx.
        for r in range(R):
            #print(f"yy size:{yy.size()},{R},{C}")
            phat_data[p].update({f"y_{p}_{r}":yhat[r]}) #no, don't; do it separately.
                # if include_nuisance:
                #     pass #unimplemented — get MLE for gamma, yuck
                # Above is commented out because we actually make this correction post-Hessian now.


    #Get hessians and sample params

    #Start with theta

    transformed_hat_data = dict()
    for k,v in chain(hat_data.items(),*[phat.items() for phat in phat_data]):
        if k in transformation:
            transformed_hat_data[k] = transformation[k](v)
        else:
            transformed_hat_data[k] = v

    real_hessian = not MINIMAL_DEBUG
    if real_hessian:
        hess_center = pyro.condition(model,transformed_hat_data)
        mytrace = poutine.block(poutine.trace(hess_center).get_trace)(data, scale, include_nuisance)
        log_posterior = mytrace.log_prob_sum()
    theta_part_names = ["sdrc", "sdprc", "ec", "erc"]
    theta_parts = []
    theta_hat_data = dict()
    for part_name in theta_part_names: #TODO: theta_hat_data redundant with hat_data, now we have phat_data????
        if part_name in hat_data:
            #print(f"adding {part_name} to theta_parts")
            theta_parts.append(hat_data[part_name]) #fails if missing (ie, eprc)
            theta_hat_data[part_name] = hat_data[part_name]
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
                    len(theta_hat_data), #tensors, not elements=tlen
                    blocksize,
                    return_grad=True)
    else:
        neg_big_hessian, big_grad = -torch.eye(full_len), torch.zeros(full_len)
    big_hessian = -neg_big_hessian #TODO: in-place
    Info = big_hessian[:tlen][:tlen]

    all_means = torch.cat([tpart.view(-1) for tpart in theta_parts],0)
    theta_mean = all_means[:tlen]

    #declare global-level psi params
    globalpsi = pyro.param('globalpsi',torch.ones(tlen)*BASE_PSI,
                constraint=constraints.positive)
    globalInfo = Info[:tlen,:tlen]
    M = infoToM(globalInfo,globalpsi)
    adjusted = globalInfo+M
    #print("matrix?",globalInfo.size(),M.size(),theta_mean.size(),adjusted.size(),[(float(globalInfo[i,i]),float(M[i,i])) for i in range(tlen)])#,np.linalg.det(adjusted))
    theta = pyrosample('theta',
                    dist.MultivariateNormal(theta_mean, precision_matrix=adjusted),
                    infer={'is_auxiliary': True})

    #decompose theta into specific values
    tmptheta = theta
    for pname, phat in theta_hat_data.items():
        elems = phat.nelement()
        pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
        #print(f"adding {pname} from theta ({elems}, {phat.size()}, {tmptheta.size()}, {pdat})" )

        if pname in transformation:
            pyrosample(pname, dist.Delta(transformation[pname](pdat.view(phat.size())))
                                .to_event(len(list(phat.size())))) #TODO: reshape after transformation, not just before???
        else:
            pyrosample(pname, dist.Delta(pdat.view(phat.size()))
                                .to_event(len(list(phat.size()))))
    with all_ps() as p_tensor:
        for p in p_tensor:
            for pname, phat in phat_data[p].items():
                elems = phat.nelement()
                pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
                #print(f"adding {pname} from theta ({elems}, {phat.size()}, {tmptheta.size()}, {pdat})" )

                if pname in transformation:
                    pyrosample(pname, dist.Delta(transformation[pname](pdat.view(phat.size())))
                                        .to_event(len(list(phat.size())))) #TODO: reshape after transformation, not just before???
                else:
                    pyrosample(pname, dist.Delta(pdat.view(phat.size()))
                                        .to_event(len(list(phat.size()))))
    assert list(tmptheta.size())[0] == 0


    combinedpsi = torch.cat([globalpsi, precinctpsi],0)


    precinct_newton_step_multiplier_logit = pyro.param(
            'precinct_newton_step_multiplier_logit',ts(0.))
    epnsml = torch.exp(precinct_newton_step_multiplier_logit)
    step_mult = MAX_NEWTON_STEP * epnsml / (1 + epnsml)
    global_indices = ts(range(tlen))
    with all_ps() as p_tensor:
        for p in p_tensor:


            precinct_indices = ts(range(tlen + p*precelems, tlen + (p+1)*precelems))

            full_indices = torch.cat([global_indices,precinct_indices],0)

            #print(f"HW:{P},{p},{B},{P//B},{len(big_HWs)},{big_HWs[p//B].size()},")
            #print(f"HW2:{big_HWs[p//B].size()},{list(full_indices)}")
            HW = big_hessian.index_select(0,full_indices).index_select(1,full_indices)


            #one step of Newton's method on the Ws
            precinct_cov = torch.inverse(HW[tlen:,tlen:])

            precinct_grad = big_grad.index_select(0,precinct_indices) #[tlen + pp*(R-1)*(C-1): tlen + (pp+1)*(R-1)*(C-1)]
            precinct_adj = step_mult * torch.mv(precinct_cov, precinct_grad)




            #print(f"size:{HW.size()},{tlen},{len(precinct_indices)},{len(combinedpsi)}")
            M = infoToM(HW,combinedpsi)
            precision = HW + M
            Sig = torch.inverse(precision) #This is not efficient computationally — redundancy.
                            #But I don't want to hand-code the right thing yet.
            #print(f"substep:{tlen},{Info.size()},{Sig.size()}")
            substep = torch.mm(Sig[tlen:tlen+(R-1)*(C-1), :tlen], adjusted)

            #print(f"wmean1:{theta.size()},{theta_mean.size()},{5+5}")
            #print(f"wmean2:{what[p].size()},{what[p].contiguous().view(-1).size()},{precinct_adj.size()},{torch.mv(substep, (theta - theta_mean)).size()}")
            wmean = (what[p].contiguous().view(-1) - precinct_adj[:(R-1)*(C-1)] +
                    torch.mv(substep, (theta - theta_mean))[:(R-1)*(C-1)])
            subSig = Sig[tlen:tlen+(R-1)*(C-1), tlen:tlen+(R-1)*(C-1)]
            subDel = torch.mm(substep, Sig[:tlen, tlen:tlen+(R-1)*(C-1)])
            #print(f"wmean3:{subSig.size()},{subDel.size()}")

            wSig = subSig - subDel
            wSig = wSig + infoToM(wSig, combinedpsi[tlen:tlen+(R-1)*(C-1)])

            if include_nuisance:
                eprc_adjusted = (precinct_adj[(R-1)*(C-1):] / step_mult).view(R,C)
                eprc = pyro.sample(f"eprc_{p}",
                    dist.Delta(eprc_adjusted).to_event(2)) #Not adjusting for deviation from hat!!!! TODO: fix!!!!

                if p==(P-1):
                    pass
                    #print(f"eprc_{p} adjusted {eprc.size()}")
            try:
                w = pyrosample(f"w_{p}",
                                dist.MultivariateNormal(wmean, wSig),
                                infer={'is_auxiliary': True})
            except:
                print(wSig)
                print(f"det:{np.linalg.det(wSig.data.numpy())}")
                raise
            indep = get_indep(R, C, ns[p], vs[p])
            #print(f"wmean3:{wmean.size()},{wSig.size()},{w.size()}")
            y = polytopize(R,C,w.view(R-1,C-1),indep)
            for r in range(R):
                yy = pyrosample(f"y_{p}_{r}", dist.Delta(y[r]).to_event(1))
    if do_print:
        go_or_nogo.printstuff2()

    print("guide:end")


nsteps = 5001
subset_size = 17
bigprime = 73 #Wow, that's big!


def get_subset(data,size,i):

    ns, vs = data
    P = len(ns)
    indices = ts([((i*size + j)* bigprime) % P for j in range(size)])
    subset = (ns.index_select(0,indices) , vs.index_select(0,indices))
    scale = sum(ns) / sum(subset[0]) #likelihood impact of a precinct proportional to n
    #print(f"scale:{scale}")
    return(subset,scale)

def trainGuide():

    # Let's generate voting data. I'm conditioning concentration=1 for numerical stability.

    data = model()
    #print(data)


    # Now let's train the guide.
    svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO())

    pyro.clear_param_store()
    losses = []
    for i in range(nsteps):
        subset, scale = get_subset(data,subset_size,i)
        print("svi.step(...",i,scale)
        loss = svi.step(subset,scale,True,do_print=(i % 10 == 0))
        losses.append(loss)
        if i % 10 == 0:
            reload(go_or_nogo)
            go_or_nogo.printstuff(i,loss)
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
    # print("yhat[0]:",
    #     polytopize(4,3,pyroStore["what_0"],
    #                get_indep(4,3,ns[0],vs[0])))

    return(svi,losses,data)
