import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.bernoulli import Bernoulli
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from optimizers import Lookahead # Local file.

'''
Activation function (important for Beta variates).
'''

def InverseLinear(x):
   # Inverse-Linear activation function. It controls
   # the gradient when using Beta variates and prevents
   # explosions.
   return 1.0 / (1.0-x+torch.abs(x)) + x + torch.abs(x)


'''
Generator
'''

class Generator(nn.Module):

   def __init__(self):
      super().__init__()

      # The 'ref' parameter will allow seamless random
      # generation on CUDA. It indirectly stores the
      # shape of 'z' but is never updated during learning.
      self.ref = nn.Parameter(torch.zeros(8))

      # Two hidden layers.
      self.hidden_layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(8, 64),
         nn.BatchNorm1d(64),
         nn.ReLU(),

         # Second hidden layer.
         nn.Linear(64, 128),
         nn.BatchNorm1d(128),
         nn.ReLU(),
      )

      # The visible layer is a Beta variate.
      self.alpha = nn.Linear(128, 312)
      self.beta  = nn.Linear(128, 312)

   def detfwd(self, z):
      '''Deterministic part of the generator.'''
      # Transform by passing through the layers.
      h = self.hidden_layers(z)
      # Get the parameters of the Beta variate.
      a = InverseLinear(self.alpha(h))
      b = InverseLinear(self.beta(h))
      return a,b

   def forward(self, nsmpl):
      zero = torch.zeros_like(self.ref) # Proper device.
      one = torch.ones_like(self.ref)   # Proper device.
      z = Normal(zero, one).sample([nsmpl])
      a,b = self.detfwd(z)
      return Beta(a,b).rsample()


'''
Discriminator.
'''

class Discriminator(nn.Module):

   def __init__(self):
      super().__init__()

      # Two hidden layers. As for the generator, batch
      # normalization helps reduce mode collapse.
      self.layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(312, 128),
         nn.BatchNorm1d(128),
         nn.ReLU(),
         nn.Dropout(p=0.3),

         # Second hidden layer.
         nn.Linear(128, 128),
         nn.BatchNorm1d(128),
         nn.ReLU(),
         nn.Dropout(p=0.3),

         # Third hidden layer.
         nn.Linear(128, 64),
         nn.BatchNorm1d(64),
         nn.ReLU(),
         nn.Dropout(p=0.3),

         # Visible layer.
         nn.Linear(64, 1),
      )

   def forward(self, x):
      return self.layers(x)


'''
Data model.
'''

class SpliceData:

   def __init__(self, path, randomize=True, test=True):
      def fmt(line):
         items = line.split()
         base = float(items[1])
         return [(float(x) + base) / 100. for x in items[2:]]
      with open(path) as f:
         self.data = [fmt(line) for line in f]
      # Create train and test data.
      if test:
         if randomize: np.random.shuffle(self.data)
         sztest = len(self.data) // 10 # 10% for the test.
         self.test = self.data[-sztest:]
         self.data = self.data[:-sztest]

   def batches(self, test=False, randomize=True, btchsz=32):
      data = self.test if test else self.data
      # Produce batches in index format (i.e. not text).
      idx = np.arange(len(data))
      if randomize: np.random.shuffle(idx)
      # Define a generator for convenience.
      for ix in np.array_split(idx, len(idx) // btchsz):
         yield torch.tensor([data[i] for i in ix])


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   genr = Generator()
   disc = Discriminator()

   #genr.load_state_dict(torch.load('genr-200.tch'))
   #disc.load_state_dict(torch.load('disc-200.tch'))

   data = SpliceData('exon_data.txt', test=False)

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda':
      genr.cuda()
      disc.cuda()
   
   lr = .00005 # The celebrated learning rate

   # Optimizer of the generator (RMSprop).
   gopt = torch.optim.RMSprop(genr.parameters(), lr=lr)

   # Optimizer of the discriminator (RMSprop).
   dopt = torch.optim.RMSprop(disc.parameters(), lr=lr)

   stop = True
   for epoch in range(1000):
      batch_is_over = False
      wdist = 0.
      batches = data.batches()
      while True:

         # PHASE I: compute Wasserstein distance.
         for _ in range(5):
            try:
               batch = next(batches)
            except StopIteration:
               batch_is_over = True
               break
            nsmpl = batch.shape[0]
            # Clamp to prevent NaNs in the log-likelihood.
            real = torch.clamp(batch, min=.001, max=.999).to(device)
            with torch.no_grad():
               fake = torch.clamp(genr(nsmpl), min=.001, max=.999)

            # Clamp weights.
            for p in disc.parameters():
               p.data.clamp_(-.01,+.01)

            loss = disc(fake).mean() - disc(real).mean()

            dopt.zero_grad()
            loss.backward()
            dopt.step()

         # This is the Wasserstein distance.
         wdist += float(-loss)

         # PHASE II: update the generator
         fake = torch.clamp(genr(nsmpl), min=.001, max=.999)

         loss = - disc(fake).sum() / nsmpl

         gopt.zero_grad()
         loss.backward()
         gopt.step()

         if batch_is_over: break

      # Display update at the end of every epoch.
      sys.stderr.write('Epoch %d, wdist: %f\n' % (epoch+1, wdist))

      # Save the networks.
      torch.save(genr.state_dict(), 'genr-%d.tch' % (epoch+1))
      torch.save(disc.state_dict(), 'disc-%d.tch' % (epoch+1))
