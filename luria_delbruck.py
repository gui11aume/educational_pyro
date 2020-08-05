#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numbers
import os
import pyro
import torch

import pyro.distributions as dist

from pyro.infer import Trace_ELBO, TraceEnum_ELBO, config_enumerate, JitTraceEnum_ELBO, SVI
from pyro.ops.indexing import Vindex
from pyro.optim import Adam

smoke_test = ('CI' in os.environ)
pyro.enable_validation()
pyro.set_rng_seed(123)

# Span up to the max, here 107.
MAXLEN = 107
X = torch.arange(1, MAXLEN+1)
prob = torch.zeros(MAXLEN+1)
prob[1:] = 1. / (X * (1. + X))

# Precompute convolutions (here up to 29).
MAXP = 29
CONV = torch.zeros(MAXLEN+1, MAXP+1)

M = torch.cat([torch.zeros(MAXLEN), prob]).unfold(0, MAXLEN+1, 1)
CONV[0,0] = 1.    # 0 mutation: 0 mutant.
CONV[:,1] = prob  # 1 mutation: standard distribution
for i in range(2,MAXP+1):
   # i mutations: compute the convolution.
   CONV[:,i] = torch.matmul(M, torch.flip(CONV[:,i-1], [0]))
# Add one more value to prevent 'Categorical' from scaling the
# probabilities (the value is otherwise not used).
CONV = torch.cat([CONV, 1-CONV.sum(dim=0, keepdim=True)], 0)


def model():
   # Half-Cauchy prior on the mutation rate.
   lmbd = pyro.sample('lmbd', dist.HalfCauchy(torch.ones(1)))
   # Truncated Poisson (up to MAXP).
   logits = torch.distributions.Poisson(lmbd).log_prob(torch.arange(MAXP+1.))
   z = pyro.sample(
         name = 'z',
         fn = dist.Categorical(logits=logits),
         infer = {'enumerate': 'parallel'}
   )
   with pyro.plate('plate_x'):
      prob = Vindex(CONV)[:,z]
      return pyro.sample(
            name = 'x',
            fn = dist.Categorical(prob)
      )

def guide():
   # Model 'lmbd' as a LogNormal variate.
   a = pyro.param('a', torch.zeros(1))
   b = pyro.param('b', torch.ones(1))
   lmbd = pyro.sample('lmbd', dist.LogNormal(a,b))
   # Truncated Poisson
   logits = torch.distributions.Poisson(lmbd).log_prob(torch.arange(MAXP+1.))
   pyro.sample(
         name = 'z',
         fn = dist.Categorical(logits=logits),
         infer = {'enumerate': 'parallel'}
   )


def serving_model(n):
   # Model 'lmbd' as a LogNormal variate.
   import pdb; pdb.set_trace()
   a = pyro.param('a', torch.zeros(1))
   b = pyro.param('b', torch.ones(1))
   lmbd = torch.distributions.log_normal.LogNormal(a,b).sample([n]).view(-1)
   # Truncated Poisson
   logits = torch.distributions.Poisson(lmbd).log_prob(torch.arange(MAXP+1.))
   z = torch.distributions.categorical.Categorical(logits=logits).sample()
   return z


# Data from the Luria-Delbruck experiment.
obs = torch.tensor([1,0,3,0,0,5,0,5,0,6,107,0,0,0,1,0,0,64,0,35])
cnd_model = pyro.poutine.condition(model, data={ 'x': obs })


optimizer = Adam({ 'lr': 0.05 })
#loss = JitTraceEnum_ELBO(max_plate_nesting=1)
loss = TraceEnum_ELBO(max_plate_nesting=1)
svi = SVI(cnd_model, guide, optimizer, loss)

l = 0
for step in range(1, 1001):
   l += svi.step()
   if step % 50 == 0:
      print(float(l))
      l = 0.

print('===')

a = pyro.param('a')
b = pyro.param('b')

print(serving_model(100))
