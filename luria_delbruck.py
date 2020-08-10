#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pyro
import torch

pyro.enable_validation()
pyro.set_rng_seed(123)

MAXP = 10
MAXLEN = 107


def precompute_convolutions(maxp, maxlen):
   # Start with the "standard" distribution. This is a granular
   # version of the number of descendents of an individual
   # picked at random during exponential growth.
   X = torch.arange(1, maxlen+1)
   prob = torch.zeros(maxlen+1)
   prob[1:] = 1. / (X * (1. + X))

   # Now precompute convolutions (up to maxp and maxlen).
   conv = torch.zeros(maxp+1, maxlen+1)

   M = torch.cat([torch.zeros(maxlen), prob]).unfold(0, maxlen+1, 1)
   # 0 mutation: 0 mutant, beyond that, compute the convolution.
   conv[0,0] = 1.
   for i in range(1, maxp+1):
      conv[i,:] = torch.matmul(M, torch.flip(conv[i-1,:], [0]))
   # Add one more value to prevent 'Categorical' from scaling the
   # probabilities (the value is otherwise not used).
   return torch.cat([conv, 1-conv.sum(dim=1, keepdim=True)], 1)

DISTR = precompute_convolutions(MAXP, MAXLEN)

# Truncated Poisson (up to and including MAXP).
poisson = (lambda lmbd: torch.distributions.Poisson(lmbd)
   .log_prob(torch.arange(MAXP+1.)).exp())


def model(x=None, lmbd=None):
   nx = x.shape[0] if x is not None else 1
   if lmbd is None:
      # Half-Cauchy prior on the mutation rate.
      lmbd = pyro.sample('lmbd', pyro.distributions.HalfCauchy(torch.ones(1)))
   probs = poisson(lmbd)
   with pyro.plate('plate_x', nx):
      # Number of mutations.
      z = pyro.sample(
            name = 'z',
            fn = pyro.distributions.Categorical(probs=probs),
            infer = {'enumerate': 'parallel'}
      )
      # Number of resistant individuals.
      x = pyro.sample(
            name = 'x',
            fn = pyro.distributions.Categorical(DISTR[z,:]),
            obs = x
      )
      return z, x


def guide(*args, **kwargs):
   # Model 'lmbd' as a LogNormal variate.
   a = pyro.param('a', torch.zeros(1))
   b = pyro.param('b', torch.ones(1),
         constraint=torch.distributions.constraints.positive)
   return pyro.sample('lmbd', pyro.distributions.LogNormal(a,b))


# Data from the Luria-Delbruck experiment (resistant individuals).
obs = torch.tensor([1,0,3,0,0,5,0,5,0,6,107,0,0,0,1,0,0,64,0,35])


optimizer = pyro.optim.Adam({ 'lr': 0.05 })
ELBO = pyro.infer.JitTraceEnum_ELBO()
svi = pyro.infer.SVI(model, guide, optimizer, ELBO)

loss = 0
for step in range(1, 501):
   loss += svi.step(obs)
   if step % 50 == 0:
      print(loss)
      loss = 0.

print('===')
print('Sampling mutation rate')
a = pyro.param('a')
b = pyro.param('b')
print(torch.distributions.log_normal.LogNormal(a,b).sample([18]).view(-1))

print('===')
print('Average mutation rate')
print(torch.distributions.log_normal.LogNormal(a,b).sample([1000]).mean())

@pyro.infer.infer_discrete(first_available_dim=-2, temperature=1)
def inference_model(x = None):
   z, x = model(x=x, lmbd=guide())
   return z

print('===')
print('Sampling mutations for 107 resistant individuals')
print(inference_model(107 * torch.ones(100)))
