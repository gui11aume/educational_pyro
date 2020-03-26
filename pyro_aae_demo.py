import numpy as np
import pyro
import pyro.distributions as dist
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.constraints as constraints

from torch.distributions.beta import Beta
from torch.distributions.normal import Normal

from aae_demo import *

# Turn on internal checks for debugging.
# Available only from Pyro 1.3.0.
try:
   pyro.enable_validation(True)
except AttributeError:
   pass

class DecoderModel:
   def __init__(self, decoder):
      self.decd = decoder

   def __call__(self, obs, idx):
      nsmpl = obs.shape[0]
      one = torch.ones_like(self.decd.ref)
      mix = 2 * Bernoulli(.2 * one[0]).sample([nsmpl]) - 1.
      mu = torch.ger(mix, one) # Array of +/-1.
      sd = one.expand([nsmpl,-1])
      # Pyro sample referenced "z" (latent variable).
      # This plate has to be present in the guide as well.
      with pyro.plate('plate_z', nsmpl):
         # Here we indicate that the rightmost dimension is
         # the shape of satistical events (4), and that we
         # want "nsmpl" independent events.
         z = pyro.sample('z', dist.Normal(mu,sd).to_event(1))
      # Push forward through the layers.
      with torch.no_grad():
         a,b = self.decd(z)
      # Remove unobserved variables.
      ax = a[:,idx]
      bx = b[:,idx]
      with pyro.plate('plate_x', nsmpl):
         # Pyro sample referenced "x" (observed variables).
         pyro.sample('x', dist.Beta(ax,bx).to_event(1), obs=obs)

class DecoderGuide:
   def __init__(self, decoder):
      self.decd = decoder

   def __call__(self, obs, idx):
      nsmpl = obs.shape[0]
      zero = torch.zeros_like(self.decd.ref).expand([nsmpl,-1])
      one  = torch.ones_like(self.decd.ref).expand([nsmpl,-1])
      mu = pyro.param('mu', zero)
      sd = pyro.param('sd', .3 * one, constraint=constraints.positive)
      # Pyro sample referenced "z" (latent variables).
      # This plate was present in the model.
      with pyro.plate('plate_z', nsmpl):
         # Here we indicate that the rightmost dimension is
         # the shape of satistical events (4), and that we
         # want "nsmpl" independent events.
         pyro.sample('z', dist.Normal(mu,sd).to_event(1))


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   decd = Decoder()
   decd.load_state_dict(torch.load('decd-200.tch'))

   data = SpliceData('exon_data.txt')

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda': decd.cuda()

   decd.eval()
   model = DecoderModel(decd)
   guide = DecoderGuide(decd)

   # Declare Adam-based Stochastic Variational Inference engine.
   adam_params = {'lr': 0.005, 'betas': (0.90, 0.999)}
   opt = pyro.optim.Adam(adam_params)
   svi = pyro.infer.SVI(model, guide, opt, loss=pyro.infer.Trace_ELBO())
   
   X = 123 # Predict variable 123.
   for batch in data.batches(btchsz=256):
      b = batch.shape[0] # Batch size.
      d = batch.shape[1] # Dimension of the observations.
      # Pre-process the data to avoid NaN on 0s and 1.
      batch = torch.clamp(batch, min=.01, max=.99).to(device)
      # Remove variable to predict.
      idx = torch.tensor(np.delete(np.arange(d), X)).to(device)
      obs = batch[:,idx]
      loss = 0
      for step in range(5000):
         loss += svi.step(obs, idx)
         if (step+1) % 100 == 0:
            print(loss)
            loss = 0
      # Inferred parameters.
      infmu = pyro.param('mu')
      infsd = pyro.param('sd')
      import pdb; pdb.set_trace()
      # Sample latent variables with approximate posterior.
      z = Normal(infmu, infsd).sample([100]).view(100*b,-1)
      # Propagate forward and sample observable 'x'.
      a,b = decd(z)
      xx = Beta(a[:,X:X+1],b[:,X:X+1]).sample().view(100,-1,1)
      x = xx.sum(dim=0) / 100.
      # Compare.
      torch.cat([x, batch[:,X:X+1]], dim=1)
