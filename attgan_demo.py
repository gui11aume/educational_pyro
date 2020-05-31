import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.bernoulli import Bernoulli
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from optimizers import Lamb, Lookahead # Local file.

'''
Activation function (important for Beta variates).
'''

def InverseLinear(x):
   # Inverse-Linear activation function. It controls
   # the gradient when using Beta variates and prevents
   # explosions.
   return 1.0 / (1.0-x+torch.abs(x)) + x + torch.abs(x)


'''
Dot product attention layer.
'''

class SelfAttentionLayer(nn.Module):

   def __init__(self, h, d_model, dropout=0.1):
      assert d_model % h == 0 # Just to be sure.
      super().__init__()
      self.h = h
      self.d = d_model

      # Linear mappings.
      self.Wq = nn.Linear(d_model, d_model)
      self.Wk = nn.Linear(d_model, d_model)
      self.Wv = nn.Linear(d_model, d_model)

      # Output layers.
      self.do = nn.Dropout(p=dropout)
      self.Wo = nn.Linear(d_model, d_model)
      self.ln = nn.LayerNorm(d_model)

   def forward(self, X):
      h  = self.h       # Number of heads.
      H  = self.d // h  # Head dimension.
      N  = X.shape[0]   # Batch size.

      # Linear transforms.
      q = self.Wq(X).view(N,h,-1).transpose(0,1)
      k = self.Wk(X).view(N,h,-1).transpose(0,1)
      v = self.Wv(X).view(N,h,-1).transpose(0,1)

      # Dot products.
      A = torch.matmul(q,  k.transpose(-2,-1))

      if N > 1:
         maskdiag = torch.eye(N, device=A.device) == 1.
         A = A.masked_fill(maskdiag, float('-inf'))

      # Attention softmax.
      p_attn = F.softmax(A, dim=-1)

      # Apply attention to v.
      Oh = torch.matmul(p_attn, v)

      # Concatenate attention output.
      O = Oh.transpose(1,2).contiguous().view_as(X)

      # Layer norm and residual connection.
      return self.ln(X + self.do(self.Wo(O)))


'''
Feed foward layer.
'''

class FeedForwardNet(nn.Module):
   def __init__(self, d_model, d_ffn, dropout=0.1):
      super().__init__()
      self.ff = nn.Sequential(
         nn.Linear(d_model, d_ffn),
         nn.ReLU(),
         nn.Linear(d_ffn, d_model)
      )
      self.do = nn.Dropout(p=dropout)
      self.ln = nn.LayerNorm(d_model)

   def forward(self, X):
      return self.ln(X + self.do(self.ff(X)))


''' 
Transformer layer.
'''

class TransformerLayer(nn.Module):
   def __init__(self, h, d_model, d_ffn, dropout=0.1):
      super().__init__()
      self.h = h
      self.d = d_model
      self.f = d_ffn
      self.sattn = SelfAttentionLayer(h, d_model, dropout=dropout)
      self.ffn   = FeedForwardNet(d_model, d_ffn, dropout=dropout)
      
   def forward(self, X, mask=None):
      return self.ffn(self.sattn(X))


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
      z = Normal(zero, one).sample([nsmpl])
      a,b = self.detfwd(z)
      return Beta(a,b).rsample()


'''
Discriminator.
'''

class Discriminator(nn.Module):

   def __init__(self):
      super().__init__()

      self.layers = nn.Sequential(
         # Embedding.
         nn.Linear(312, 512),

         # Transformer.
         TransformerLayer(8, 512, 1024),

         # First fully connected layer.
         nn.Linear(512, 256),
         nn.BatchNorm1d(256),
         nn.ReLU(),
         nn.Dropout(p=0.3),

         # Second fully connected layer.
         nn.Linear(256, 64),
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

   def batches(self, test=False, randomize=True, btchsz=128):
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

   genr.load_state_dict(torch.load('genr-att-140.tch'))
   disc.load_state_dict(torch.load('disc-att-140.tch'))

   data = SpliceData('exon_data.txt', test=False)

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda':
      genr.cuda()
      disc.cuda()
   
   lr = 0.0001 # The celebrated learning rate

#   # Optimizer of the generator (Lookahead with Lamb).
#   gbase = Lamb(genr.parameters(),
#         lr=lr, weight_decay=0.01, betas=(.9, .999), adam=True)
#   gopt = Lookahead(base_optimizer=gbase, k=5, alpha=0.8)

   # Optimizer of the generator (SGD)
   gopt = torch.optim.SGD(genr.parameters(), lr=10*lr)

   # Optimizer of the discriminator (Lookahead with Lamb). 
   dbase = Lamb(disc.parameters(),
         lr=lr, weight_decay=0.01, betas=(.9, .999), adam=True)
   dopt = Lookahead(base_optimizer=dbase, k=5, alpha=0.8)

   # (Binary) cross-entropy loss.
   loss_fun = nn.CrossEntropyLoss(reduction='mean')

   for epoch in range(200):
      dloss = floss = 0.
      for batch in data.batches():
         import pdb; pdb.set_trace()
         nsmpl = batch.shape[0]

         # Prevent NaNs in the log-likelihood.
         batch = torch.clamp(batch, min=.01, max=.99).to(device)

         # PHASE I: update the discriminator.
         zero = torch.zeros(nsmpl).long().to(device)
         real = batch
         loss = loss_fun(disc(real), zero)

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
      torch.save(genr.state_dict(), 'genr-att-%d.tch' % (epoch+1))
      torch.save(disc.state_dict(), 'disc-att-%d.tch' % (epoch+1))

   # Done, save the networks.
   torch.save(genr.state_dict(), 'genr-att-%d.tch' % (epoch+1))
   torch.save(disc.state_dict(), 'disc-att-%d.tch' % (epoch+1))
