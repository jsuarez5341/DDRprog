from pdb import set_trace as T
import torch as t
from torch import nn

class RandomOptim:
   def __init__(self, net, lr=1e-2, cuda=True):
      self.net = net
      self.params = [e for e in net.parameters()]
      self.lr = lr
      self.grads = [[] for _ in range(len(self.params))]
      self.cuda = cuda

   def step(self):
      for ind, param in enumerate(self.params):
         grad = self.lr * t.randn(*(param.size()))
         if self.cuda:
            grad = grad.cuda()
         param.data += grad
         self.grads[ind] = grad

   def reset(self):
      for ind, param in enumerate(self.params):
         param.data -= self.grads[ind]
         self.grads[ind] *= 0
      
