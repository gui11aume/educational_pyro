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
      self.ref = nn.Parameter(torch.zeros(4))

      # Two hidden layers.
      self.hidden_layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(4, 64),
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

   def forward(self, nsmpl, return_z=False):
      zero = torch.zeros_like(self.ref) # Proper device.
      one = torch.ones_like(self.ref)   # Proper device.
#      one = torch.ones_like(self.ref)
#      mix = 2 * Bernoulli(.2 * one[0]).sample([nsmpl]) - 1.
#      mu = torch.ger(mix, one) # Array of +/-1.
#      sd = one.expand([nsmpl,-1])
#      z = Normal(mu, sd).sample()
      z = Normal(zero, one).sample([nsmpl])
      a,b = self.detfwd(z)
      if return_z: return z, Beta(a,b).rsample()
      else:        return Beta(a,b).rsample()


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
         nn.Linear(128, 64),
         nn.BatchNorm1d(64),
         nn.ReLU(),
         nn.Dropout(p=0.3),

         # Visible layer.
         nn.Linear(64, 2),
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

#   genr.load_state_dict(torch.load('genr-200.tch'))
#   disc.load_state_dict(torch.load('disc-200.tch'))

   data = SpliceData('exon_data.txt', test=False)

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda':
      genr.cuda()
      disc.cuda()
   
   lr = 0.001 # The celebrated learning rate

#   # Optimizer of the generator (Adam).
   gbase = torch.optim.Adam(genr.parameters(), lr=lr,
         weight_decay=0.01, betas=(.9, .999))
   gopt = Lookahead(base_optimizer=gbase, k=5, alpha=0.8)

   # Optimizer of the discriminator (Adam).
   dbase = torch.optim.Adam(disc.parameters(), lr=lr,
         weight_decay=0.01, betas=(.9, .999))
   dopt = Lookahead(base_optimizer=dbase, k=5, alpha=0.8)

#   gopt = torch.optim.SGD(genr.parameters(), lr=lr)
#   dopt = torch.optim.SGD(disc.parameters(), lr=lr)

   # (Binary) cross-entropy loss.
   loss_fun = nn.CrossEntropyLoss(reduction='mean')

   for epoch in range(200):
      dloss = floss = 0.
      for batch in data.batches():
#         import pdb; pdb.set_trace()
         nsmpl = batch.shape[0]

         # Prevent NaNs in the log-likelihood.
         batch = torch.clamp(batch, min=.01, max=.99).to(device)

         # PHASE I: update the discriminator.
         zero = torch.zeros(nsmpl).long().to(device)
         real = batch
         loss = loss_fun(disc(real), zero)

#         # DEBUG
#         with torch.no_grad():
#            z, fake = genr(nsmpl, return_z=True)
#            idx = (fake.sum(1) / 312) > .5
#            np.savetxt(sys.stdout, z[idx,].cpu().numpy(), fmt='%.3f')
#         continue
#         # END DEBUG

         one = torch.ones(nsmpl).long().to(device)
         fake = torch.clamp(genr(nsmpl), min=.01, max=.99)
         loss += loss_fun(disc(fake), one)

         dopt.zero_grad()
         loss.backward()
         dopt.step()

         dloss += float(loss)

         # PHASE II: update the generator
         zero = torch.zeros(2*nsmpl).long().to(device)
         fake = torch.clamp(genr(2*nsmpl), min=.01, max=.99)
         loss = loss_fun(disc(fake), zero)

         gopt.zero_grad()
         loss.backward()
         gopt.step()

         floss += float(loss)

      # Display update at the end of every epoch.
      sys.stderr.write('Epoch %d, disc: %f, fool: %f\n' % \
            (epoch+1, dloss, floss))
      # DEBUG
      torch.save(genr.state_dict(), 'genr-%d.tch' % (epoch+1))
      torch.save(disc.state_dict(), 'disc-%d.tch' % (epoch+1))

   # Done, save the networks.
   torch.save(genr.state_dict(), 'genr-%d.tch' % (epoch+1))
   torch.save(disc.state_dict(), 'disc-%d.tch' % (epoch+1))
