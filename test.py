import numpy as np
import pyro
import pyro.distributions as dist
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.beta import Beta

# Turn on internal checks for debugging.
# Available only from Pyro 1.3.0.
try:
   pyro.enable_validation(True)
except AttributeError:
   pass

lin = nn.Linear(2, 100)
lin.cuda()

def model(obs, device='cuda'):
   nsmpl = obs.shape[0]
   D2 = torch.zeros(2).to(device)
   zero = torch.zeros_like(D2)
   one = torch.ones_like(D2)
   # Pyro sample referenced "z" (latent variable).
   # This plate has to be present in the guide as well.
   with pyro.plate('plate_z', nsmpl):
      z = pyro.sample('z', dist.Normal(zero,one).to_event(1))
   mu = lin(z)
   # Pyro sample referenced "x" (observed variables).
   with pyro.plate('plate_x'):
      pyro.sample('x', dist.Normal(mu, .1).to_event(1), obs=obs)

def guide(obs, device='cuda'):
   nsmpl = obs.shape[0]
   D2 = torch.zeros(2).to(device)
   zero = torch.zeros_like(D2)
   one  = torch.ones_like(D2)
   mu = pyro.param('mu', zero)
   sd = pyro.param('sd', one)
   # Pyro sample referenced "z" (latent variables).
   # This plate was present in the model.
   with pyro.plate('plate_z', nsmpl):
      # here we indicate that the rightmost dimension is
      # the shape of satistical events (8), and that we
      # want "nsmpl" independent events.
      pyro.sample('z', dist.Normal(mu,sd).to_event(1))


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   # Declare an Adam-based Stochastic Variational Inference engine.
   adam_params = {"lr": 0.01, "betas": (0.90, 0.999)}
   opt = pyro.optim.Adam(adam_params)
   svi = pyro.infer.SVI(model, guide, opt, loss=pyro.infer.Trace_ELBO())
   
   obs = Normal(lin(torch.tensor([[2.,-2.]]).to(device)), .1).sample()

   loss = 0.
   for step in range(1000):
      loss += svi.step(obs)
      if (step+1) % 100 == 0:
         print(loss)
         loss = 0
   import pdb; pdb.set_trace()
   mu = pyro.param('mu')
   sd = pyro.param('sd')
      
