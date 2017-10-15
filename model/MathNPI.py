from pdb import set_trace as T
import torch as t
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np

from lib import utils
from model.ConvLSTM import ConvLSTMCell
from model import ExecutionEngine
from model.RHNCell import RHNCell

class Iden(nn.Module):
   def __init__(self):
      super().__init__()
   
   def forward(self, x):
      return x

class ConvReluNorm(nn.Module):
   def __init__(self, h, k):
      super(ConvReluNorm, self).__init__()
      self.conv = utils.Conv2d(h, h, k)
      self.norm = nn.InstanceNorm2d(h)

   def forward(self, x):
      return self.norm(F.relu(self.conv(x)))

class ReluNorm(nn.Module):
   def __init__(self, h):
      super(ReluNorm, self).__init__()
      self.norm = nn.InstanceNorm2d(h)

   def forward(self, x):
      return self.norm(F.relu(x))

class UnaryModule(nn.Module):
   def __init__(self, h):
      super(UnaryModule, self).__init__()
      self.fc1 = nn.Linear(h, h)

   def forward(self, x):
      return self.fc1(x)

class BinaryModule(nn.Module):
   def __init__(self, h):
      super(BinaryModule, self).__init__()
      self.weight = Parameter(t.Tensor(1, 4))
      #self.reset_parameters()

      self.fc0 = nn.Linear(2*h, h, bias=False)
      self.fc1 = nn.Linear(h, h, bias=False)
      self.fc2 = nn.Linear(h, h, bias=False)
      self.fc3 = nn.Linear(h, h, bias=False)
      self.fc4 = nn.Linear(h, h, bias=False)

   def reset_parameters(self):
      stdv = 1. / np.sqrt(self.weight.size(1))
      self.weight.data.uniform_(-stdv, stdv)

   def forward(self, x1, x2):
      x1 = self.fc1(x1)
      x2 = self.fc2(x2)
      xx  = t.stack([x1+x2, x1-x2, x1*x2, x1/(x2+1e-4)], 1)
      xx = t.sum(xx * self.weight, 1)
      return xx
      '''
      xx = t.cat((x1, x2))
      xx = self.fc0(xx)
      xx = F.relu(xx)
      xx = self.fc1(xx)
      return xx
      xx = xx.view(2, -1)
      x1, x2 = xx[0], xx[1]
      a = self.fc1(x1+x2)
      b = self.fc2(x1-x2)
      c = self.fc3(x1*x2)
      d = self.fc4(x1/(x2+1e-4))
      return (a+b+c+d)
      '''
      #norm = t.sqrt(t.sum(t.abs(self.weight)))
      #weight = self.weight / norm
      #ret = F.sigmoid(self.weight) * xx
      ret = t.sum(ret, 1).view(1, 1)
      return ret

class NPILSTM(nn.Module):
   def __init__(self, embedDim, h, probLen, probVocab, aVocab):
      super().__init__()
      self.lstmCell = nn.LSTMCell(embedDim, h)
      self.embed = nn.Embedding(probVocab, embedDim)
      self.probProj = nn.Linear(h, probVocab)
      self.ansProj  = nn.Linear(h, 1)
      self.embedDim = embedDim
      self.h = h
   
   def forward(self, x, trainable=False):
      prob, ans = x
      prob = prob.long()
      embed = self.embed(prob)

      batch, seq = prob.size()
      numUnary = (seq-2)//2
      hInit = Variable(1e-2*t.randn(batch, self.h)).cuda()
      cInit = Variable(1e-2*t.randn(batch, self.h)).cuda()
      hc = (hInit, cInit)
      preds = []
      for i in range(seq-1):
         xi = embed[:, i]
         h, c = self.lstmCell(xi, hc)
         h = h.view(batch, self.h)
         c = c.view(batch, self.h)
         hc = (h, c)

         ansPreds  = self.ansProj(h)
         if i > numUnary+1:
            preds += [ansPreds]

      ansPreds = t.cat(preds, 1)
      return ansPreds

class AnsModule(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.fc1 = nn.Linear(h, h)
      self.fc2 = nn.Linear(h, 100)

   def forward(self, x):
      return self.fc2(F.relu(self.fc1(x)))

def eqnAns(eqn):
   class eqnNum:
      def __init__(self, val):
         self.val = val

   stack = []
   for tok in eqn:
      tok = tok.data[0]
      if tok < 3:
         pass
      elif tok < 13:
         stack.append(tok-3)
      else:
         arg2 = stack.pop()
         arg1 = stack.pop()
         if tok == 13:
            ret = arg1 + arg2
         elif tok == 14:
            ret = arg1 - arg2
         elif tok == 15:
            ret = arg1 * arg2
         elif tok == 16:
            ret = arg1 / arg2
         else:
            T()
         if ret is None:
            return None
         stack.append(ret)

   return stack.pop()

class NPICells(nn.Module):
   def __init__(self, embedDim, h, 
            pVocab, numUnary, numBinary): 
      super().__init__()
      pVocab = 10
      self.embed = nn.Embedding(pVocab, embedDim)
      #self.probProj = nn.Linear(h, pVocab)
      #self.ansProj  = AnsModule(h)
      self.embedDim = embedDim
      self.h = h
      self.pVocab = pVocab
      self.numUnary = numUnary
      self.numBinary = numBinary

      #self.fork = UnaryModule(h)
      #prefix   = [Iden()]
      binaries = [BinaryModule(h) for i in range(4)]
      #binaries = [lambda x1, x2: x1+x2,
      #            lambda x1, x2: x1-x2,
      #            lambda x1, x2: x1*x2,
      #            lambda x1, x2: x1/x2]
      #self.cells = binaries
      self.cells = nn.ModuleList(binaries)
      self.proj = nn.Linear(embedDim, 1)
      self.lstmCell = nn.LSTMCell(embedDim, h)
  
   def forward(self, x, trainable=False):
      prob, ans = x
      prob = prob.long()
      if not trainable:
         ans = None

      batch, seq = prob.size()
      hInit = Variable(1e-2*t.randn(batch, self.h)).cuda()
      cInit = Variable(1e-2*t.randn(batch, self.h)).cuda()
      hc = (hInit, cInit)

      stack   = [[] for i in range(batch)]
      outMat = [[] for i in range(batch)]
      for i in range(1, seq-1):
         outList = []
         for j in range(batch):
            cellInd = prob[j, i]
            cellInt = cellInd.data[0]
            if cellInt <= self.numUnary:
               #out = (cellInd - 3).float()/10.0
               out = self.embed(cellInd-3) 
            else:
               arg2 = stack[j].pop()
               arg1 = stack[j].pop()
               cell = self.cells[cellInt-13]
               out = cell(arg1, arg2)
               #outMat[j] += [out]

            outList.append(out)

         out = t.stack(outList, 0)
         h, c = self.lstmCell(out, hc)
         h = h.view(batch, self.h)
         c = c.view(batch, self.h)
         hc = (h, c)
         out = h

         for j in range(batch):
            cellInd = prob[j, i]
            cellInt = cellInd.data[0]
            outj = out[j]
            if cellInt > self.numUnary:
               outMat[j] += [outj]
            stack[j].append(outj)


      if i == seq-2:
         out = t.stack([t.stack(e, 0) for e in outMat], 0)
         batch, seqLen, oLen = out.size()
         out = out.view(-1, self.embedDim)
         ansPreds = self.proj(out).view(batch, seqLen)
         
      return ansPreds


