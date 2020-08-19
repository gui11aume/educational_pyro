#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pyro
import torch

pyro.enable_validation()
pyro.set_rng_seed(123)

MAXP = 100
MAXLEN = 303


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
poisson = (lambda lmbd: torch.distributions.Poisson(lmbd.view(-1,1))
   .log_prob(torch.arange(MAXP+1.)).exp())


def model(scale, x=None, lmbd=None):
   if x is not None:
      assert x.shape == scale.shape
   nx = x.shape[0] if x is not None else 1
   if lmbd is None:
      # Half-Cauchy prior on the mutation rate.
      base_rate = torch.ones(1) * 1e-9
      lmbd = pyro.sample('lmbd', pyro.distributions.HalfCauchy(base_rate))
   probs = poisson(lmbd * scale)
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
      return lmbd, z, x


def guide(*args, **kwargs):
   # Model 'lmbd' as a LogNormal variate.
   a = pyro.param('a', -20.0 * torch.ones(1))
   b = pyro.param('b',   0.2 * torch.ones(1),
         constraint=torch.distributions.constraints.positive)
   return pyro.sample('lmbd', pyro.distributions.LogNormal(a,b))


# Data from the Luria-Delbruck experiment.
# Size of the cultures.
scale = torch.cat([
   torch.ones(9)  * 3.4e10,                         # Experiment 1
   torch.ones(8)  * 4.0e10,                         # Experiment 10
   torch.ones(10) * 4.0e10,                         # Experiment 11
   torch.ones(10) * 2.9e10,                         # Experiment 15
   torch.ones(20) * 5.6e8,                          # Experiment 16
   torch.ones(12) * 5.0e8,                          # Experiment 17
   torch.ones(19) * 1.1e8,                          # Experiment 21a
   torch.ones(5)  * 3.2e10,                         # Experiment 21b
])
# Resistant individuals.
x = torch.tensor([
   10,18,125,10,14,27,3,17,17,                      # Experiment 1
   29,41,17,20,31,30,7,17,                          # Experiment 10
   30,10,40,45,183,12,173,23,57,51,                 # Experiment 11
   6,5,10,8,24,13,165,15,6,10,                      # Experiment 15
   1,0,3,0,0,5,0,5,0,6,107,0,0,0,1,0,0,64,0,35,     # Experiment 16
   1,0,0,7,0,303,0,0,3,48,1,4,                      # Experiment 17
   0,0,0,0,8,1,0,1,0,15,0,0,19,0,0,17,11,0,0,       # Experiment 21a
   38,28,35,107,13                                  # Experiment 21b
])


optimizer = pyro.optim.Adam({ 'lr': 0.05 })
ELBO = pyro.infer.JitTraceEnum_ELBO(max_plate_nesting=1)
svi = pyro.infer.SVI(model, guide, optimizer, ELBO)

loss = 0
for step in range(1, 1001):
   loss += svi.step(scale, x)
   if step % 100 == 0:
      print(loss)
      loss = 0.

print('===')
print('Sampling mutation rate')
a = pyro.param('a')
b = pyro.param('b')
print(torch.distributions.log_normal.LogNormal(a,b).sample([18]).view(-1))

print('===')
print('Average mutation rate')
smpl = torch.distributions.log_normal.LogNormal(a,b).sample([10000]).view(-1)
print(smpl.mean())

print('===')
print('99% credible interval')
lo = torch.kthvalue(smpl, 50).values
hi = torch.kthvalue(smpl, 9951).values
print(torch.tensor([lo, hi]))

@pyro.infer.infer_discrete(first_available_dim=-2)
def inference_model(scale, x=None):
   return model(scale, x=x, lmbd=guide())

print('===')
print('Sampling mutations for 125 resistant individuals (Expt 1)')
mutations = list()
for _ in range(100):
   lmbd, z, x = inference_model(torch.tensor([3.4e10]), torch.tensor([125]))
   mutations.append(z)
print(torch.stack(mutations))

print('Sampling mutations for 3 resistant individuals (Expt 1)')
mutations = list()
for _ in range(100):
   lmbd, z, x = inference_model(torch.tensor([3.4e10]), torch.tensor([3]))
   mutations.append(z)
print(torch.stack(mutations))

print('Sampling mutations for 107 resistant individuals (Expt 16)')
mutations = list()
for _ in range(100):
   lmbd, z, x = inference_model(torch.tensor([5.6e8]), torch.tensor([3]))
   mutations.append(z)
print(torch.stack(mutations))