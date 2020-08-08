#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numbers
import os
import pyro
import torch

import pyro.distributions as dist

from pyro.infer import Trace_ELBO, TraceEnum_ELBO, JitTraceEnum_ELBO, SVI
from pyro.optim import Adam

smoke_test = ('CI' in os.environ)
pyro.enable_validation()
pyro.set_rng_seed(123)


def model():
   # Half-Cauchy prior on the mutation rate.
   lmbd = pyro.sample('lmbd', dist.HalfCauchy(torch.ones(1)))
   with pyro.plate('plate_x'):
      return pyro.sample(
            name = 'x',
            fn = dist.Bernoulli(1.-torch.exp(-lmbd)),
            infer = {'enumerate': 'parallel'}
      )
      
def guide():
   # Model 'lmbd' as a LogNormal variate.
   a = pyro.param('a', torch.zeros(1))
   b = pyro.param('b', torch.ones(1),
         constraint=pyro.distributions.constraints.positive)
   lmbd = pyro.sample('lmbd', dist.LogNormal(a,b))


# Data from the Luria-Delbruck experiment.
obs = torch.tensor([1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0,0,1,0,1]).float()
cnd_model = pyro.poutine.condition(model, data={ 'x': obs })


optimizer = Adam({ 'lr': 0.05 })
ELBO = JitTraceEnum_ELBO(max_plate_nesting=1)
svi = SVI(cnd_model, guide, optimizer, ELBO)

l = 0
for step in range(1, 101):
   l += svi.step()
   if step % 5 == 0:
      print(float(l))
      l = 0.

print('===')

a = pyro.param('a')
b = pyro.param('b')

#print(torch.distributions.log_normal.LogNormal(a,b).sample([100]).view(-1))
print(torch.distributions.log_normal.LogNormal(a,b).sample([5]).view(-1))
print(torch.distributions.log_normal.LogNormal(a,b).sample([1000]).view(-1).mean())
