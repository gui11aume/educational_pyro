#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pyro
import torch

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
DEFINITION OF MODEL AND GUIDE.
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


def guide(scale, x=None, mask=None):
   # Model 'lmbd' as a LogNormal variate.
   a = pyro.param('a', -20.0 * torch.ones(1, device=scale.device))
   b = pyro.param('b',   0.2 * torch.ones(1, device=scale.device),
         constraint=torch.distributions.constraints.positive)
   with pyro.plate('plate_lmbd', scale.shape[0]):
      lmbd = pyro.sample('lmbd', pyro.distributions.LogNormal(a,b))
   return lmbd


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

optimizer = pyro.optim.Adam({ 'lr': .01 })
ELBO = pyro.infer.JitTraceEnum_ELBO(max_plate_nesting=2,
      ignore_jit_warnings=True)
#ELBO = pyro.infer.TraceEnum_ELBO(max_plate_nesting=2)
svi = pyro.infer.SVI(model, guide, optimizer, ELBO)



if __name__ == '__main__':

   loss = 0
   for step in range(1, 1001):
      loss += svi.step(scale=scale, x=x, mask=mask)
      if step % 100 == 0:
         print(loss)
         loss = 0.


   # Run the predictive model to see the quality of the fit.
   print('===')
   print('Running predictive model')
   predictive = pyro.infer.Predictive(model=model, guide=guide,
         num_samples=1000)
   with torch.no_grad():
      predsmpl = predictive(scale, mask=mask)


   print('===')
   print('Average mutation rate and 90% credible interval')
   smpl_lmbd = predsmpl.get('lmbd').view(-1)
   print(smpl_lmbd.mean())
   lo = torch.kthvalue(smpl_lmbd, k=400).values
   hi = torch.kthvalue(smpl_lmbd, k=7600).values
   print(torch.tensor([lo, hi]))


   print('===')
   print('Average sum per experiment and 90% credible interval')
   smpl_x = predsmpl.get('x')
   smpl_s = (smpl_x * mask).sum(dim=-1)
   print((x * mask).sum(dim=-1))
   print(smpl_s.sum(dim=0) / 10000.)
   lo = torch.kthvalue(smpl_s, k=50, dim=0).values
   hi = torch.kthvalue(smpl_s, k=950, dim=0).values
   print(torch.stack([hi, lo]))


   print('===')
   print('Sampling mutations for 125 resistant individuals (Expt 1)')
   scale_ = torch.tensor([[3.4e10]]).expand([100,1])
   x_ = torch.tensor([[125]]).expand([100,1])
   # Faster and simpler than 'infer_discrete'.
   ELBO = pyro.infer.discrete.TraceEnumSample_ELBO(max_plate_nesting=2)
   with torch.no_grad():
      ELBO.loss(model=model, guide=guide, scale=scale_, x=x_)
   lmbd, z, x = ELBO.sample_saved()
   print(z.view(-1))

