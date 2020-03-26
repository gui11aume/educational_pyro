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
Encoder-Decoder
'''

class Encoder(nn.Module):

   def __init__(self):
      super().__init__()

      # The 'ref' parameter will allow seamless random
      # generation on CUDA. It indirectly stores the
      # shape of 'z' but is never updated during learning.
      self.ref = nn.Parameter(torch.zeros(2))

      # Three hidden layers.
      self.hidden_layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(312, 128),
         nn.LayerNorm(128),
         nn.ReLU(),

         # Second hidden layer.
         nn.Linear(128, 64),
         nn.LayerNorm(64),
         nn.ReLU(),

         # Welcome to the latent space.
         nn.Linear(64, 2),
      )

   def rnd(self, nsmpl):
      '''The distribution to match (Gaussian mixture).'''
      one = torch.ones_like(self.ref)  # On the proper device.
      mix = 2 * Bernoulli(.2*one[0]).sample([nsmpl]) - 1.
      return Normal(torch.ger(mix,one), one).sample()

   def forward(self, x):
      return self.hidden_layers(x)


class Decoder(nn.Module):

   def __init__(self):
      super().__init__()

      # The 'ref' parameter will allow seamless random
      # generation on CUDA. It indirectly stores the
      # shape of 'z' but is never updated during learning.
      self.ref = nn.Parameter(torch.zeros(2))

      # Two hidden layers.
      self.hidden_layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(2, 64),
         nn.LayerNorm(64),
         nn.ReLU(),

         # Second hidden layer.
         nn.Linear(64, 128),
         nn.LayerNorm(128),
         nn.ReLU(),
      )

      # The visible layer is a Beta variate.
      self.alpha = nn.Linear(128, 312)
      self.beta  = nn.Linear(128, 312)

   def forward(self, z):
      # Transform by passing through the layers.
      h = self.hidden_layers(z)
      # Get the parameters of the Beta variate.
      a = InverseLinear(self.alpha(h))
      b = InverseLinear(self.beta(h))
      return a,b


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
         nn.Linear(2, 64),
         nn.LayerNorm(64),
         nn.ReLU(),
         nn.Dropout(p=0.3),

         # Second hidden layer.
         nn.Linear(64, 128),
         nn.LayerNorm(128),
         nn.ReLU(),
         nn.Dropout(p=0.3),

         # Visible layer.
         nn.Linear(128, 2),   
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

   encd = Encoder()
   decd = Decoder()
   disc = Discriminator()

   encd.load_state_dict(torch.load('encd-200.tch'))
   decd.load_state_dict(torch.load('decd-200.tch'))
   disc.load_state_dict(torch.load('disc-200.tch'))

   data = SpliceData('exon_data.txt', test=False)

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda':
      encd.cuda()
      decd.cuda()
      disc.cuda()
   
   lr = 0.001 # The celebrated learning rate

   # Optimizer of the encoder (Adam).
   abase = torch.optim.Adam(encd.parameters(), lr=lr,
         weight_decay=0.01, betas=(.9, .999))
   aopt = Lookahead(base_optimizer=abase, k=5, alpha=0.8)

   # Optimizer of the decoder (Adam).
   bbase = torch.optim.Adam(decd.parameters(), lr=lr,
         weight_decay=0.01, betas=(.9, .999))
   bopt = Lookahead(base_optimizer=bbase, k=5, alpha=0.8)

   # Optimizer of the discriminator (Adam).
   cbase = torch.optim.Adam(disc.parameters(), lr=lr,
         weight_decay=0.01, betas=(.9, .999))
   copt = Lookahead(base_optimizer=cbase, k=5, alpha=0.8)

   # (Binary) cross-entropy loss.
   loss_fun = nn.CrossEntropyLoss(reduction='mean')

   nbtch = 0
   for epoch in range(200):
      dloss = floss = closs = 0.
      for batch in data.batches():
         import pdb; pdb.set_trace()
         nbtch += 1
         nsmpl = batch.shape[0]

         # Prevent NaNs in the log-likelihood.
         batch = torch.clamp(batch, min=.01, max=.99).to(device)

         # PHASE I: update the discriminator.
         zero = torch.zeros(nsmpl).long().to(device)
         disc_loss = loss_fun(disc(encd.rnd(nsmpl)), zero)

         # Then we make a batch of positive (fake) cases.
         with torch.no_grad():
            z = encd(batch)
         one = torch.ones(nsmpl).long().to(device)
         disc_loss += loss_fun(disc(z), one)

         copt.zero_grad()
         disc_loss.backward()
         copt.step()

         dloss += float(disc_loss)

         # PHASE II: update the encoder-decoder
         z = encd(batch)
         a,b = decd(z)

         fool_loss = loss_fun(disc(z), zero)
         cstr_loss = -Beta(a,b).log_prob(batch).sum() / batch.numel()

         floss += float(fool_loss)
         closs += float(cstr_loss)

         loss = fool_loss + cstr_loss

         aopt.zero_grad()
         bopt.zero_grad()
         loss.backward()
         aopt.step()
         bopt.step()


      # Display update at the end of every epoch.
      sys.stderr.write('Epoch %d, disc: %f, fool: %f, cstr: %f\n' % \
            (epoch+1, dloss, floss, closs))

   # Done, save the networks.
   torch.save(encd.state_dict(), 'encd-%d.tch' % (epoch+1))
   torch.save(decd.state_dict(), 'decd-%d.tch' % (epoch+1))
   torch.save(disc.state_dict(), 'disc-%d.tch' % (epoch+1))
