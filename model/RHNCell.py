from pdb import set_trace as T

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from lib import utils

def highwayGate(Ws, s, dropout=None):
   trainable = dropout is not None
   h = int(Ws.size()[1]/2)
   hh, tt  = t.split(Ws, h, 1)
   hh, tt = F.tanh(hh), F.sigmoid(tt)
   cc = 1 - tt
   tt = F.dropout(tt, p=dropout, training=trainable)
   return hh*tt + s*cc

class RHNCell(nn.Module):

   def __init__(self, xDim, hDim, depth=1):
      super().__init__()
      self.depth = depth
      
      self.input  = nn.Linear(xDim, 2*hDim)
      self.hidden = utils.list(nn.Linear, hDim, 2*hDim, n=depth)

   def forward(self, x, s=None, trainable=False):
      for i in range(self.depth):
         Ws = self.hidden[i](s) if s is not None else 0
         if i == 0:
            Ws += self.input(x)
         s = highwayGate(Ws, s, trainable)
      return s
