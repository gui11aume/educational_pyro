#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pyro
import torch

assert pyro.__version__.startswith('1.4.0')

pyro.enable_validation()
pyro.set_rng_seed(123)

MAXP = 100
MAXLEN = 303


# Run on GPU if we can.
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


'''
LURIA-DELBRUCK PROBABILITY DISTRIBUTION.
'''

def precompute_convolutions(maxp, maxlen):
   # Start with the "standard" distribution. This is a less
   # granular version of the number of descendents of an
   # individual picked at random during exponential growth.
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

STDISTR = precompute_convolutions(MAXP, MAXLEN).to(device)


# Truncated Poisson (up to and including MAXP).
def poisson(lmbd, device):
   parm = torch.clamp(lmbd.unsqueeze(-1), max=MAXP)
   pois = torch.distributions.Poisson(parm)
   return pois.log_prob(torch.arange(MAXP+1., device=device)).exp()



'''
DEFINITION OF MODEL.
'''

def model(scale, x=None, mask=None):
   if x is not None:
      assert x.shape == scale.shape
   # Half-Cauchy prior on the mutation rate lambda ('lmbd').
   with pyro.plate('plate_lmbd', scale.shape[0]):
      # Scale the prior to the right order of
      # magnitude for the performed experiments.
      base_rate = torch.ones(1, device=scale.device) * 1e-9
      lmbd = pyro.sample('lmbd', pyro.distributions.HalfCauchy(base_rate))
   probs = poisson(lmbd.unsqueeze(-1) * scale, scale.device)
   # Nested design.
   plate_repl = pyro.plate('plate_replicate', scale.shape[1])
   plate_expt = pyro.plate('plate_experiment', scale.shape[0])
   with plate_repl, plate_expt:
      if mask is None:
         mask = torch.ones_like(scale)
      # Number of mutations.
      z = pyro.sample(
            name = 'z',
            fn = pyro.distributions.Categorical(probs=probs).mask(mask),
            infer = {'enumerate': 'parallel'}
      )
      # Number of resistant individuals.
      x = pyro.sample(
            name = 'x',
            fn = pyro.distributions.Categorical(STDISTR[z,:]).mask(mask),
            obs = x
      )
   return lmbd, z, x



'''
DATA FROM THE LURIA-DELBRUCK EXPERIMENT.
'''

# Data from the Luria-Delbruck experiment.
# Size of the cultures.
scale = torch.cat([
  3.4e10  * torch.ones(20),    # Experiment 1
  4.0e10  * torch.ones(20),    # Experiment 10
  4.0e10  * torch.ones(20),    # Experiment 11
  2.9e10  * torch.ones(20),    # Experiment 15
  5.6e08  * torch.ones(20),    # Experiment 16
  5.0e08  * torch.ones(20),    # Experiment 17
  1.1e08  * torch.ones(20),    # Experiment 21a
  3.2e10  * torch.ones(20),    # Experiment 21b
]).view(8,20).to(device)

# Resistant individuals.
_ = 0 # Padding value. 
x = torch.tensor([
   [10,18,125,10,14,27,3,17,17,    _,_,_,_,_,_,_,_,_,_,_],
   [29,41,17,20,31,30,7,17,      _,_,_,_,_,_,_,_,_,_,_,_],
   [30,10,40,45,183,12,173,23,57,51, _,_,_,_,_,_,_,_,_,_],
   [6,5,10,8,24,13,165,15,6,10,      _,_,_,_,_,_,_,_,_,_],
   [1,0,3,0,0,5,0,5,0,6,107,0,0,0,1,0,0,64,0,35         ],
   [1,0,0,7,0,303,0,0,3,48,1,4,          _,_,_,_,_,_,_,_],
   [0,0,0,0,8,1,0,1,0,15,0,0,19,0,0,17,11,0,0,         _],
   [38,28,35,107,13,       _,_,_,_,_,_,_,_,_,_,_,_,_,_,_]
]).to(device)

# Masking (to pass data as tensor).
mask = torch.tensor([
   [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
   [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
   [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
   [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
   [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
   [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
]).to(device)



'''
INFERENCE.
'''


if __name__ == '__main__':

   # NUTS kernel with JIT compilation (for speed).
   kernel = pyro.infer.mcmc.NUTS(model, adapt_step_size=True,
         max_plate_nesting=2, jit_compile=True, ignore_jit_warnings=True,
         init_strategy=pyro.infer.autoguide.initialization.init_to_sample)
   # Run the MCMC with 2 chains (each size 2000 after 300 warm up iterations).
   mp_context = 'forkserver' if device == 'cpu' else 'spawn'
   mcmc = pyro.infer.mcmc.MCMC(kernel, num_samples=1000, warmup_steps=300,
         num_chains=2, mp_context='forkserver')
   # Feed in the experimental data.
   mcmc.run(scale=scale, x=x, mask=mask)
   smpl_lmbd = mcmc.get_samples().get('lmbd')
   lo = torch.kthvalue(smpl_lmbd, k=50, dim=0).values
   hi = torch.kthvalue(smpl_lmbd, k=950, dim=0).values
   print(torch.stack([hi, lo]))
