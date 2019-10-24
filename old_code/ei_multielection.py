#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
print('Yes, I will run.')

from importlib import reload
import contextlib
from itertools import chain
import cProfile as profile
import inspect
from collections import OrderedDict, defaultdict


from matplotlib import pyplot as plt
import numpy as np

import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro import poutine


from utilities import myhessian
from utilities.rank1torch import optimize_Q
from utilities import go_or_nogo
from utilities.cmult import CMult
from utilities import polytopize
reload(polytopize)
from utilities.polytopize import get_indep, polytopize, depolytopize, to_subspace, process_data
from utilities.posdef import *

ts = torch.tensor


torch.manual_seed(478301986) #Gingles

pyro.enable_validation(True)
pyro.set_rng_seed(0)



init_narrow = 10  # Numerically stabilize initialization.


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
MAX_NEWTON_STEP = .7 #currently, just taking this much of a step, hard-coded
EPRCHAT_HESSIAN_POINT_FRACTION = .5
RECENTER_PRIOR_STRENGTH = 2.



NSTEPS = 5001
SUBSET_SIZE = 5
BIG_PRIME = 73 #Wow, that's big!

MINIMAL_DEBUG = False
if MINIMAL_DEBUG:
    pyrosample = lambda x,y,infer=None : y.sample()
else:
    pyrosample = pyro.sample

FACTOR_NAMES = ["t",#time (year)
                "o",#office (pres/sen/gov)
                ]

FACTOR_SIZES = [3, 3] #3 years, 3 offices; check data later

class GlobalTerm:
    def __init__(self, hasRace, hasFacs):
        self.hasRace = hasRace
        self.hasFacs = hasFacs
        self.setName()

    def setName(self):
        name = "alpha_p" #party, always
        if self.hasRace:
            name += "o"
        for (hasFac, facName) in zip(self.hasFacs, FACTOR_NAMES):
            if hasFac:
                name += facName
        self.name = name
        self.signame = name + "_sigma"

    def getSigmaName(self,):

    def modelSample(self,):
        pass

    def modelApply(self,):
        pass

    def guideParam(self,):
        pass

    def guideToModel(self,):
        pass

    def modelPregen(self,):
        pass

class LocalTerm(GlobalTerm)

def model(data=None, scale=1., include_nuisance=False, do_print=False):
    print("model:begin",scale,include_nuisance)
    if data is None:
        P, R, C = 30, 4, 3
        P, R, C = 30, 3, 2
        ns = torch.zeros(P,R)
        for p in range(P):
            for r in range(R):
                ns[p,r] = 20 * (p*r - 5*(r+1-R) + 6) + ((p-15)*(r-1))^2
                ns[p,r] = 20 * (p*r - 5*(r+1-R) + 6) + ((p-1)*(r-1))^2

                    # pyrosample('precinctSizes',
                    #         dist.NegativeBinomial(p*r - 5*(r+1-R) + 6, .95))
    else:
        ns, vs, indeps, tots, factors, facNums, covs = data
        assert len(ns)==len(vs)
        # Hyperparams.
        P = len(ns)
        R = len(ns[0])
        C = len(vs[0])
        nFactors = factors.size()[1]
        assert len(facNums) == nFactors
        nCovs = covs.size()[1]

    prepare_ps = range(P)
    ps_plate = pyro.plate('all_sampled_ps',P)
    @contextlib.contextmanager
    def all_sampled_ps():
        with ps_plate as p, poutine.scale(scale=scale) as pscale:
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
    #param_residual=pyrosample('param_residual', dist.Normal(0.,1.))

    ec = pyrosample('ec', dist.Normal(torch.zeros(C),sdc).to_event(1))
    erc = pyrosample('erc', dist.Normal(torch.zeros(R,C),sdrc).to_event(2))
    if include_nuisance:
        with all_sampled_ps() as p_tensor:
            logits = (
                pyrosample(f'logits', dist.Normal((ec + erc).expand(P,R,C), sdprc).to_event(2))
                ) #eprc.size() == [P,R,C] because plate dimension happens on left
        #print("Model: sampling eprc",eprc[0,0,0])
    else:
        logits = torch.zeros(P,R,C) #dummy for print statements. TODO:remove

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

    with all_sampled_ps() as p_tensor:#pyro.plate('precinctsm2', P):
        if data is None:
            y = torch.zeros(P,R,C)
            for p in p_tensor:
                for r in range(R):
                    tmp = dist.Multinomial(int(ns[p,r]),logits=logits[p,r]).sample()
                    #print(f"y_{p}_{r}: {tmp} {y[p,r]}")
                    y[p,r] = tmp
        else:
            if not torch.all(torch.isfinite(logits)):
                print("logits!!!")
                for p in range(P):
                    if not torch.all(torch.isfinite(logits[p])):
                        print("nan in logits[p]",p)
                        print(logits[p])
                        print(ec)
                        print(erc)
            y = pyro.sample(f"y",
                        CMult(1000,logits=logits).to_event(1))
                        #dim P, R, C from plate, to_event, CMult
                        #note that n is totally fake — sums are what matter.
                        #TODO: fix CMult so this fakery isn't necessary.
            print("model y",scale,y.size(),y[0,:2,:2])
            #print("ldim",logits.size(),y[0,0,0])

    if data is None:
        #
        print(f"ec:{ec}")
        print(f"erc:{erc}")
        print(f"y[0]:{y[0]}")
        vs = torch.sum(y,1)

        return (ns,vs)

    print("model:end")



def recenter_rc(rc):
    rowcentered= (rc - torch.mean(rc,0))
    colcentered = rowcentered - torch.mean(rowcentered,0)
    return colcentered


def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno


# This function is to guess a good starting point for eprchat. But removing for now because:
# 1. SIMPLIFY!
# 2. We're doing 1 (partial) step of Newton's Method anyway so this is mostly redundant.


# def initial_eprc_hat_guess(totsp,pi_p,Q2,Q_precision,pi_precision):
#     pi_precision = totsp / pi_p / (torch.ones_like(pi_p) - pi_p)
#     pweighted_Q = pi_p + (Q2-pi_p) * Q_precision / (Q_precision + pi_precision)
#     return (torch.log(pweighted_Q/pi_p))

def guide(data, scale, include_nuisance=False, do_print=False):
    print("guide:begin",scale,include_nuisance)

    ns, vs, indeps, tots = data
    P = len(ns)
    R = len(ns[1])
    C = len(vs[1])

    prepare_ps = range(P) #for dealing with hatted quantities (no pyro.sample)
    ps_plate = pyro.plate('all_sampled_ps',P)
    @contextlib.contextmanager
    def all_sampled_ps(): #for dealing with unhatted quantities (include pyro.sample)
        with ps_plate as p, poutine.scale(scale=scale) as pscale:
            yield p

    #Start with hats.

    theta_hat_data = OrderedDict()
    fhat_data = OrderedDict() #frequentist
    phat_data = OrderedDict()
    transformation = defaultdict(lambda: lambda x: x) #factory of identity functions

    logsdrchat = pyro.param('logsdrchat',ts(-1.))
    fhat_data.update(sdrc=logsdrchat)
    transformation.update(sdrc=torch.exp)
    if include_nuisance:
        logsdprchat = pyro.param('logsdprchat',ts(-1.))
        fhat_data.update(sdprc=logsdprchat)
        transformation.update(sdprc=torch.exp)
        eprchat_startingpoint = torch.zeros(P,R,C,requires_grad =True) #not a pyro param...
        #eprchat_startingpoint[p].requires_grad_(True) #...so we have to do this manually

    echat = pyro.param('echat', torch.zeros(C))
    erchat = pyro.param('erchat', torch.zeros(R,C))
    theta_hat_data.update(ec=echat,erc=erchat)
    transformation.update(erc=recenter_rc)

    eprchat_hessian_point_fraction = EPRCHAT_HESSIAN_POINT_FRACTION #pyro.param("eprchat_fraction",ts(0.5),constraint=constraints.interval(0.0, 1.0))
    # with pyro.plate('candidatesg', C):
    #     echat = pyro.param('echat', ts(0.))
    #     with pyro.plate('rgroupsg', R):
    #         erchat = pyro.param('erchat', ts(0.))
    #         if include_nuisance:
    #             with pyro.plate('precinctsg', P):
    #                 eprchat = pyro.param('eprchat', ts(0.))
    #             theta_hat_data.update(eprc=eprchat)


    ec2 = echat.detach().requires_grad_()
    erc2 = erchat.detach().requires_grad_()

    if include_nuisance:
        sdprc2 = logsdprchat.detach().requires_grad_()

    #Including recenter_rc makes erc not identifiable
    #We could  "impose a prior" as follows, but instead, we'll just allow things to wander off.
    # recentering_amount = (RECENTER_PRIOR_STRENGTH *
    #         torch.mean((erchat - recenter_rc(erchat))**2)/torch.exp(logsdrchat))
    # pyrosample('param_residual', dist.Delta(recentering_amount))


    #Amortize hats
    yhat = []
    what = [] #P  * ((R-1) * (C-1)
    yhat2 = [] #reconstituted as a function of what
    eprchats = [] #P  * (R * C)

    logittotals = ec2 + erc2
    #print("sizes1 ",P,R,C,eprc.size(),ec.size(),erc.size(),logittotals.size())
    if include_nuisance:
        #print("adding starting point")
        logittotals = logittotals + eprchat_startingpoint # += doesn't work here because mumble in-place mumble shape
    else:
        logittotals = logittotals.expand(P,R,C)
    pi_raw = torch.exp(logittotals)
    pi = pi_raw / torch.sum(pi_raw,-1).unsqueeze(-1)



    #print("guide:pre-p")
    if include_nuisance:
        Q_precision = torch.exp(-sdprc2) * R / (R-1) #TODO: check that correction for R TODO: check that sign

    for p in prepare_ps:#pyro.plate('precinctsg2', P):
        #precalculation - logits to pi


        #get ŷ^(0)
        Q, iters = optimize_Q(R,C,pi[p],vs[p]/torch.sum(vs[p]),ns[p]/torch.sum(ns[p]),tolerance=.01,maxiters=3)
        #print(f"optimize_Q {p}:{iters}")
        yhat.append(Q*tots[p])
        if p==0:
            pass
            #print("p0", Q,tots[p])

        #depolytopize
        what.append(depolytopize(R,C,yhat[p],indeps[p]))

        yhat2.append(polytopize(R,C,what[p],indeps[p]))

        QbyR = Q/torch.sum(Q,-1).unsqueeze(-1)
        logresidual = torch.log(QbyR / pi[p])
        eprchat = logresidual * eprchat_hessian_point_fraction

        #get ν̂^(0)
        if include_nuisance:
            eprchats.append(eprchat)
            #was: initial_eprc_hat_guess(tots[p],pi[p],Q2,Q_precision,pi_precision))


    phat_data.update(y=torch.cat([y.unsqueeze(0) for y in yhat2],0))
    #print("y is",phat_data["y"].size(),yhat[-1].size(),yhat[0][0,0],yhat2[0][0,0])
    if include_nuisance:
        if False: #dummy eprchats — testing
            eprchats = [torch.zeros(R,C,requires_grad=True) for i in range(P)]
        eprchat = torch.cat([eprc.unsqueeze(0) for eprc in eprchats],0)
        logits = echat + erchat + eprchat
        phat_data.update(logits=logits)


    #print("guide:post-p")


    ################################################################################
    #Get hessians
    ################################################################################

    #Start with theta

    transformed_hat_data = OrderedDict()
    for k,v in chain(theta_hat_data.items(),phat_data.items(),fhat_data.items()):
        if k in transformation:
            transformed_hat_data[k] = transformation[k](v)
        else:
            transformed_hat_data[k] = v

    #
    #print("line ",lineno())
    hess_center = pyro.condition(model,transformed_hat_data)
    #print("line ",lineno())
    mytrace = poutine.block(poutine.trace(hess_center).get_trace)(data, scale, include_nuisance)
    #print("line ",lineno(),P,R,C)
    log_posterior = mytrace.log_prob_sum()
    print("lp: ",log_posterior,lineno())

    if True:
        yyy = transformed_hat_data["y"]

        print("guide y",yyy.size(),yyy[0,:2,:2])
        yhess = myhessian.hessian(log_posterior, yyy)
        print("yhess",yhess[:4,:4])
        whess = myhessian.hessian(log_posterior, what)
        print("whess", whess)

    if False: #test perturbing y and logits
        second_transformed_hat_data = dict(transformed_hat_data)
        for vartochange in ["y","y","logits","logits"]:
            second_transformed_hat_data = dict(second_transformed_hat_data)

            second_hess_center = pyro.condition(model,second_transformed_hat_data)
            #print("line ",lineno())
            second_mytrace = poutine.block(poutine.trace(second_hess_center).get_trace)(data, scale, include_nuisance)
            #print("line ",lineno(),P,R,C)
            second_log_posterior = second_mytrace.log_prob_sum()
            print("re-lp: ",second_log_posterior,lineno())

            new_logits = second_transformed_hat_data[vartochange]
            new_logits[0,0,0].add_(torch.ones([]))
            new_logits[0,-1,-1].add_(-torch.ones([]))
            second_transformed_hat_data[vartochange] = new_logits

            second_hess_center = pyro.condition(model,second_transformed_hat_data)
            #print("line ",lineno())
            second_mytrace = poutine.block(poutine.trace(second_hess_center).get_trace)(data, scale, include_nuisance)
            #print("line ",lineno(),P,R,C)
            second_log_posterior = second_mytrace.log_prob_sum()
            print("lp2: ",second_log_posterior,lineno())



    hessian_hats_in_sampling_order = [] #this will have eprchats — good for taking hessian but wrong for mean
    mean_hats_in_sampling_order = [] #this will have logits — right mean, but not upstream, so can't use for hessian
    for part_name in list(theta_hat_data.keys()):
        #print(f"adding {part_name} to hats_in_sampling_order")
        hessian_hats_in_sampling_order.append(theta_hat_data[part_name])
        mean_hats_in_sampling_order.append(theta_hat_data[part_name])
    theta_dims = sum(theta_part.numel() for theta_part in theta_hat_data.values())
    #print("len theta",len(hats_in_sampling_order),hats_in_sampling_order)

    #add phat_data to hats_in_sampling_order — but don't get it from phat_data because it comes in wrong format
    if include_nuisance:
        tensors_per_unit = 2 #tensors, not elements=(R-1)*(C-1) + R*C
        dims_per_unit = (R-1)*(C-1) + R*C
        for w,eprc,logit in zip(what,
                        eprchats, #This is NOT the center of the distribution; that's `logits`
                                #However, the Hessian wrt `eprchats` should equal that wrt `logits`
                                #And using `logits` here wouldn't work because it's not upstream on the autodiff graph
                        logits):
            hessian_hats_in_sampling_order.extend([w,eprc])
            mean_hats_in_sampling_order.extend([w,logit])
    else:
        tensors_per_unit = 1 #tensors, not elements=(R-1)*(C-1)
        dims_per_unit = (R-1)*(C-1)
        hessian_hats_in_sampling_order.extend(what) #`what` is already a list, not a tensor, so this works
        mean_hats_in_sampling_order.extend(what) #`what` is already a list, not a tensor, so this works
    full_dims = sum(hat.numel() for hat in mean_hats_in_sampling_order) #doesn't matter which one, hessian_... would be fine

    #print("lp:::",log_posterior)
    big_arrow, big_grad = myhessian.arrowhead_hessian_precision(log_posterior,
                    hessian_hats_in_sampling_order, #This is NOT the center of the distribution; that's `logits`
                            #However, the Hessian wrt `eprchats` should equal that wrt `logits`
                            #And using `logits` here wouldn't work because it's not upstream on the autodiff graph
                    len(theta_hat_data), #tensors, not elements=theta_dims
                    tensors_per_unit,
                    return_grad=True)


    if False: #debug arrowhead_hessian_precision
        raw_hess = myhessian.hessian(log_posterior,
                        hessian_hats_in_sampling_order)
        print("raw_hess",raw_hess[:4,:4])
        print(raw_hess[theta_dims:theta_dims+4,theta_dims:theta_dims+4])


    #declare global-level psi params
    globalpsi = pyro.param('globalpsi',torch.ones(theta_dims)*BASE_PSI,
                constraint=constraints.positive)
    #declare precinct-level psi params
    precinctpsi = pyro.param('precinctpsi',BASE_PSI * torch.ones(dims_per_unit),
                constraint=constraints.positive)

    big_arrow.setpsis(globalpsi,precinctpsi)
    big_arrow.weights = [scale] * P

    #head_precision, head_adjustment,  = rescaledSDD(-neg_big_hessian,combinedpsi) #TODO: in-place


    #theta_info = big_hessian[:theta_dims,:theta_dims]

    all_means = torch.cat([tpart.contiguous().view(-1) for tpart in mean_hats_in_sampling_order],0)
    #print("all_means",all_means)
    #print(torch.any(torch.isnan(torch.diag(neg_big_hessian))),torch.any(torch.isnan(torch.diag(big_hessian))))
    theta_mean = all_means[:theta_dims]
    #print("detirminants",np.linalg.det(theta_info.detach()),np.linalg.det(big_hessian.detach()))
    #print(theta_info[:3,:3])
    #print(-neg_big_hessian[:6,:3])

    theta = pyrosample('theta',
                    dist.MultivariateNormal(theta_mean, precision_matrix=big_arrow.marginal_gg()),
                    infer={'is_auxiliary': True})
    g_delta = theta - theta_mean

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
    # with all_sampled_ps() as p_tensor:
    #     for p in p_tensor:
    #         for pname, phat in phat_data[p].items():
    #             elems = phat.nelement()
    #             pdat, tmptheta = tmptheta[:elems], tmptheta[elems:]
    #             #print(f"adding {pname} from theta ({elems}, {phat.size()}, {tmptheta.size()}, {pdat})" )
    #
    #             if pname in transformation:
    #                 pyrosample(pname, dist.Delta(transformation[pname](pdat.view(phat.size())))
    #                                     .to_event(len(list(phat.size())))) #TODO: reshape after transformation, not just before???
    #             else:
    #                 pyrosample(pname, dist.Delta(pdat.view(phat.size()))
    #                                     .to_event(len(list(phat.size()))))
    assert list(tmptheta.size())[0] == 0


    for k,v in fhat_data.items():
        pyrosample(k, dist.Delta(transformation[k](v)))



    #TODO: uncomment the following, which is fancy logic for learning what fraction of a Newton's method step to take

    #precinct_newton_step_multiplier_logit = pyro.param(
    #        'precinct_newton_step_multiplier_logit',ts(0.))
    #epnsml = torch.exp(precinct_newton_step_multiplier_logit)
    step_mult = MAX_NEWTON_STEP #* epnsml / (1 + epnsml)
    ysamps = []
    logit_samps = []
    global_indices = ts(range(theta_dims))
    for p in range(P):


        precinct_indices = ts(range(theta_dims + p*dims_per_unit, theta_dims + (p+1)*dims_per_unit))

        #print(f"theta_1p_hess:{P},{p},{B},{P//B},{len(big_HWs)},{big_HWs[p//B].size()},")
        #print(f"HW2:{big_HWs[p//B].size()},{list(full_indices)}")
        conditional_mean, conditional_cov = big_arrow.conditional_ll_mcov(p,g_delta,
                                all_means.index_select(0,precinct_indices))

        precinct_cov = big_arrow.llinvs[p] #for Newton's method, not for sampling

        precinct_grad = big_grad.index_select(0,precinct_indices) #[theta_dims + pp*(R-1)*(C-1): theta_dims + (pp+1)*(R-1)*(C-1)]

        #print("precinct:::",theta_1p_hess.size(),precinct_cov.size(),big_grad.size(),precinct_grad.size(),)
        adjusted_mean = conditional_mean + step_mult * torch.mv(precinct_cov, precinct_grad)
                                 #one (partial, as defined by step_mult) step of Newton's method
                                 #Note: this applies to both ws and nus (eprcs). I was worried about whether that was circular logic but I talked with Mira and we both think it's actually principled.




        try:
            pstuff = pyrosample(f"pstuff_{p}",
                            dist.MultivariateNormal(adjusted_mean, conditional_cov),
                            infer={'is_auxiliary': True})

        except:
            print("error sampling pstuff",p,conditional_cov.size())
            print(conditional_cov)
            rawp = big_arrow.raw_lls[p]
            print(rawp)
            print(f"""dets:{np.linalg.det(conditional_cov.data.numpy())},
                    {np.linalg.det(rawp.data.numpy())},
                    {np.linalg.det(big_arrow.gg.data.numpy())},
                    {np.linalg.det(big_arrow._mgg.data.numpy())},""")
            print("mean",adjusted_mean.size())
            raise
        w_raw = pstuff[:(R-1)*(C-1)]
        y = polytopize(R,C,w_raw.view(R-1,C-1),indeps[p])
        ysamps.append(y.view(1,R,C))

        if include_nuisance:
            logit = pstuff[(R-1)*(C-1):].view(1,R,C)
            logit_samps.append(logit)

    with all_sampled_ps():
        ys = torch.cat(ysamps,0)
        if not torch.all(torch.isfinite(ys)):
            for p in range(P):
                if not torch.all(torch.isfinite(ys[p,:,:])):
                    print("nan in ys for precinct",p)
                    print(ys[p,:,:])
                    if include_nuisance:
                        print(logit_samps[p])
                    #
                    print("ns",ns[p])
                    print("vs",vs[p])
                    print("echat",echat)
                    print("erchat",erchat)
        pyro.sample("y", dist.Delta(ys).to_event(2))

        if include_nuisance:
            logit_samp_tensor = torch.cat(logit_samps,0)
            if not torch.all(torch.isfinite(logit_samp_tensor)):
                print("logits!!")
                for p in range(P):
                    if not torch.all(torch.isfinite(logit_samp_tensor[p,:,:])):
                        print("nan in nus for precinct",p)
                        print(logit_samp_tensor[p])
                        print(ysamps[p])
                        print("ns",ns[p])
                        print("vs",vs[p])
                        print("echat",echat)
                        print("erchat",erchat)
            pyro.sample("logits", dist.Delta(logit_samp_tensor).to_event(2))



    #ec2 = echat.detach().requires_grad_()
    #erc2 = erchat.detach().requires_grad_()

    def fix_ec_grad():
        echat.grad = echat.grad + ec2.grad
        #print("mode_hat.grad",mode_hat.grad)
    echat.fix_grad = fix_ec_grad

    def fix_erc_grad():
        erchat.grad = erchat.grad + erc2.grad
        #print("mode_hat.grad",mode_hat.grad)
    erchat.fix_grad = fix_erc_grad

    #no sdprc2 fix_grad. That's deliberate; the use of sdprc is an ugly hack and that error shouldn't bias the estimate of sdprc


    #TODO:
        #TODO:
            #TODO:
                #TODO: return value!
    #TODO:
        #TODO:
            #TODO:
                #TODO:  return value!
                    #TODO:
                        #TODO:
                            #TODO:
                                #TODO: return value!
                    #TODO:
                        #TODO:
                            #TODO:
                                #TODO:  return value!







    if do_print:
        go_or_nogo.printstuff2()

    print("guide:end")




def get_subset(data,size,i):

    ns, vs, indeps, tots = data
    P = len(ns)
    indices = ts([((i*size + j)* BIG_PRIME) % P for j in range(size)])
    subset = (ns.index_select(0,indices) , vs.index_select(0,indices),
                [indeps[p] for p in indices], [tots[p] for p in indices])
    scale = torch.sum(ns) / torch.sum(subset[0]) #likelihood impact of a precinct proportional to n
    #print(f"scale:{scale}")
    print("scale",scale)
    return(subset,scale)

def trainGuide():

    # Let's generate voting data. I'm conditioning concentration=1 for numerical stability.

    data = model()
    processed_data = process_data(data) #precalculate independence points
    #print(data)


    # Now let's train the guide.
    svi = SVI(model, guide, ClippedAdam({'lr': 0.005}), Trace_ELBO())

    pyro.clear_param_store()
    losses = []
    for i in range(NSTEPS):
        subset, scale = get_subset(processed_data,SUBSET_SIZE,i)
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
