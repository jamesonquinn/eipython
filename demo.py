#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function


from matplotlib import pyplot as plt
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
import pandas as pd
ts = torch.tensor

pyro.enable_validation(True)
pyro.set_rng_seed(0)

def model1(N,effects,errors,scalehyper,tailhyper): #groups, subgroups, groupsize by trial, options
    #prior on μ_τ
    meanEffect = pyro.sample('meanEffect',dist.Normal(ts(0.),ts(20.)))

    #prior on σ_τ
    scale = pyro.sample('scale',dist.Normal(ts(0.),torch.abs(scalehyper)))

    #prior on (df-1)
    tail = pyro.sample('tail',dist.LogNormal(ts(2.),tailhyper))

    #Latent true values
    truth = pyro.sample('truth',dist.StudentT(tail+ts(1.),meanEffect.expand(N),scale).to_event(1))

    #Observations conditional on truth (likelihood)
    observations = pyro.sample('observations', dist.Normal(truth,errors).to_event(1), obs=effects)

def model2(N,effects,errors,scalehyper,tailhyper): #groups, subgroups, groupsize by trial, options
    #prior on μ_τ
    meanEffect = pyro.sample('meanEffect',dist.Normal(ts(0.),ts(20.)))

    #prior on sd(τ)
    scale = pyro.sample('scale',dist.Normal(ts(0.),torch.abs(scalehyper)))

    #prior on muln
    tail = pyro.sample('tail',dist.LogNormal(ts(2.),tailhyper))

    sigln = torch.sqrt(ts(1.) + ts(2.) * tail * scale) - 1
    deltaln = torch.exp(tail + sigln)

    #Latent true values (offset)
    truth = pyro.sample('truth',dist.LogNormal(tail.expand(N),sigln).to_event(1))

    #Observations conditional on truth (likelihood)
    observations = pyro.sample('observations', dist.Normal(meanEffect+truth-deltaln,errors).to_event(1), obs=effects)

def guide1(N,effects,errors,scalehyper,tailhyper):
    #For the parameter of primary interest — the true mean — we fit both mean and error
    meanHat = pyro.param("meanHat",ts(0.)) #mode of the posterior for μ_τ
    meanSig = pyro.param("meanSig",ts(1.), constraint=constraints.positive)

    #For less-interesting parameters, we start by fitting just a point estimate
    logScaleHat = pyro.param("logScaleHat",ts(0.))
    logTailHat = pyro.param("logTailHat",ts(0.))

    #For latent values, we fit global posterior mean; correlation with relevant mean parameters; and variance.
    #Global mean
    truthHat = pyro.param("truthHat",effects)
    #If the mode of the posterior for
    truthRho = pyro.param("truthRho",ts(.5), constraint=constraints.unit_interval)
    truthSEmult = pyro.param("truthSEmult",ts(.9), constraint=constraints.unit_interval)

    #sample values
    meanEffect = pyro.sample('meanEffect',dist.Normal(meanHat,meanSig))
    scale = pyro.sample('scale',dist.Delta(torch.exp(logScaleHat)))
    tail = pyro.sample('tail',dist.Delta(torch.exp(logTailHat)))

    truth = pyro.sample('truth',dist.Normal(truthHat + truthRho+(meanEffect-meanHat),truthSEmult * errors).to_event(1))


def guide2(N,effects,errors,scalehyper,tailhyper):
    #
    meanHat = pyro.param("meanHat",ts(0.))
    meanSig = pyro.param("meanSig",ts(1.), constraint=constraints.positive)
    logScaleHat = pyro.param("logScaleHat",ts(0.))
    logTailHat = pyro.param("logTailHat",ts(0.))
    logTailSig = pyro.param("logTailSig",ts(1.), constraint=constraints.positive)

    truthHat = pyro.param("truthHat",effects)
    truthRho = pyro.param("truthRho",ts(.5), constraint=constraints.unit_interval)
    truthSEmult = pyro.param("truthSEmult",ts(.9), constraint=constraints.unit_interval)

    #sample values
    meanEffect = pyro.sample('meanEffect',dist.Normal(meanHat,meanSig))
    scale = pyro.sample('scale',dist.Delta(torch.exp(logScaleHat)))
    tail = pyro.sample('tail',dist.LogNormal(logTailHat,logTailSig))

    truth = pyro.sample('truth',dist.Normal(truthHat + truthRho+(meanEffect-meanHat),truthSEmult * errors).to_event(1))

data = pd.read_csv('effects_errors.csv')
effects = torch.tensor(data.effects)
errors = torch.tensor(data.errors)
N = len(effects)

def trainGuide():
    svi = SVI(model1, guide1, ClippedAdam({'lr': 0.005}), Trace_ELBO())

    pyro.clear_param_store()
    losses = []
    for i in range(3001):
        loss = svi.step(N,effects,errors,ts(10.),ts(10.))
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
