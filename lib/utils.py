import numpy as np
import sys
from pdb import set_trace as T
import time
from collections import defaultdict

import torch as t
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import torch.nn.init as init

from lib import nlp

#Generic
def invertDict(x):
   return {v: k for k, v in x.items()}

def loadDict(fName):
   with open(fName) as f:
      s = eval(f.read())
   return s

def norm(x, n=2):
   return (np.sum(np.abs(x)**n)**(1.0/n)) / np.prod(x.shape)

#Tracks inds of a permutation
class Perm():
   def __init__(self, n):
      self.inds = np.random.permutation(np.arange(n))
      self.m = n
      self.pos = 0

   def next(self, n):
      assert(self.pos + n < self.m)
      ret = self.inds[self.pos:(self.pos+n)]
      self.pos += n
      return ret

#Continuous moving average
class CMA():
   def __init__(self, n=2):
      self.t = 0.0
      self.cma = [0.0]*n
   
   def update(self, x):
      for i in range(len(x)):
         self.cma[i] = (x[i] + self.t*self.cma[i])/(self.t+1)
      self.t += 1.0

#Exponentially decaying average
class EDA():
   def __init__(self, k=0.99):
      self.k = k 
      self.eda = None
   
   def update(self, x):
      #self.eda = self.eda * k / (x * (1-k))
      if self.eda is None:
         self.eda = x
      else:
         self.eda = (1-self.k)*x + self.k*self.eda

#Print model size
def modelSize(net): 
   params = 0 
   for e in net.parameters(): 
      params += np.prod(e.size()) 
   params = int(params/1000) 
   print("Network has ", params, "K params")  

#Same padded (odd k)
def Conv2d(fIn, fOut, k, padding='same'):
   if padding == 'same':
      pad = int((k-1)/2)
   elif padding == 'valid':
      pad = 0
   return nn.Conv2d(fIn, fOut, k, padding=pad)

#ModuleList wrapper
def list(module, *args, n=1):
   return nn.ModuleList([module(*args) for i in range(n)])

#Variable wrapper
def var(xNp, volatile=False, cuda=False):
   x = Variable(t.from_numpy(xNp), volatile=volatile)
   if cuda:
      x = x.cuda()
   return x

#Full-network initialization wrapper
def initWeights(net, scheme='orthogonal'):
   print('Initializing weights. Warning: may overwrite sensitive bias parameters (e.g. batchnorm)')
   for e in net.parameters():
      if scheme == 'orthogonal':
         if len(e.size()) >= 2:
            init.orthogonal(e)
      elif scheme == 'normal':
         init.normal(e, std=1e-2)
      elif scheme == 'xavier':
         init.xavier_normal(e)

class ErrorBreakdown:
   def __init__(self):
      vocab, inv = nlp.buildVocab('data/vocab/ProgramVocab.txt')
      self.classNameDict = inv
      self.correct = defaultdict(int)
      self.total = defaultdict(int)

   def update(self, keys, correct):
      vals = [self.classNameDict[k] for k in keys]
      n = len(correct)
      for i in range(n):
         val, cor = vals[i], correct[i]
         if cor:
            self.correct[vals[i]] += 1
         self.total[vals[i]] += 1

   @property
   def scores(self):
      new = {}
      for k in self.correct.keys():
         new[k[2:]] = (self.correct[k], self.total[k])

      groups = {
      'exist': 'exist'.split(), 
      'count': 'count'.split(), 
      'compare_integer': 'equal_integer less_than greater_than'.split(), 
      'query': 'query_size query_color query_material query_shape'.split(), 
      'compare_attr': 'equal_size equal_color equal_material equal_shape'.split()
      }

      groupAccs = {}
      for groupKey in groups.keys():
         group = groups[groupKey]
         groupAcc = [0, 0]
         for k in group:
            cor, tot = new[k] 
            groupAcc[0] += cor
            groupAcc[1] += tot
         groupAccs[groupKey] = groupAcc[0] / float(groupAcc[1])

      for k in groupAccs.keys():
         print(k, ': ', groupAccs[k])

      print()
      for k in new.keys():
         new[k] = new[k][0] / float(new[k][1])
            

      order = 'exist count equal_integer less_than greater_than query_size query_color query_material query_shape equal_size equal_color equal_material equal_shape'.split(' ')
      for k in order:
         if k in new.keys():
            print(k, ': ', new[k])
      return new

class SaveManager():
   def __init__(self, root):
      self.tl, self.ta, self.vl, self.va = [], [], [], []
      self.root = root
      self.stateDict = None 
      self.lock = False

   def update(self, net, tl, ta, vl, va):
      nan = np.isnan(sum([t.sum(e) for e in net.state_dict().values()]))
      if nan or self.lock:
         self.lock = True
         print('NaN in update. Locking. Call refresh() to reset')
         return

      if self.epoch() == 1 or va[1] > np.max([e[1] for e in self.va]):
         self.stateDict = net.state_dict().copy()
         t.save(net.state_dict(), self.root+'weights')

      self.tl += [tl]; self.ta += [(ta)]
      self.vl += [vl]; self.va += [(va)]
 
      np.save(self.root + 'tl.npy', self.tl)
      np.save(self.root + 'ta.npy', self.ta)
      np.save(self.root + 'vl.npy', self.vl)
      np.save(self.root + 'va.npy', self.va)

   def load(self, net, raw=False):
      stateDict = t.load(self.root+'weights')
      self.stateDict = stateDict
      if not raw:
         net.load_state_dict(stateDict)
      self.tl = np.load(self.root + 'tl.npy').tolist()
      self.ta = np.load(self.root + 'ta.npy').tolist()
      self.vl = np.load(self.root + 'vl.npy').tolist()
      self.va = np.load(self.root + 'va.npy').tolist()

   def refresh(self, net):
      self.lock = False
      net.load_state_dict(self.stateDict)

   def epoch(self):
      return len(self.tl)+1

#From Github user jihunchoi
def _sequence_mask(sequence_length, max_len=None):
   if max_len is None: max_len = sequence_length.data.max()
   batch_size = sequence_length.size(0)
   seq_range = t.range(0, max_len - 1).long()
   seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
   seq_range_expand = Variable(seq_range_expand)
   if sequence_length.is_cuda:
      seq_range_expand = seq_range_expand.cuda()
   seq_length_expand = (sequence_length.unsqueeze(1)
                       .expand_as(seq_range_expand))
   return seq_range_expand < seq_length_expand

#From Github user jihunchoi
def maskedCE(logits, target, length):
   """
   Args:
       logits: A Variable containing a FloatTensor of size
           (batch, max_len, num_classes) which contains the
           unnormalized probability for each class.
       target: A Variable containing a LongTensor of size
           (batch, max_len) which contains the index of the true
           class for each corresponding step.
       length: A Variable containing a LongTensor of size (batch,)
           which contains the length of each data in a batch.

   Returns:
       loss: An average loss value masked by the length.
   """

   # logits_flat: (batch * max_len, num_classes)
   logits_flat = logits.view(-1, logits.size(-1))
   # log_probs_flat: (batch * max_len, num_classes)
   log_probs_flat = F.log_softmax(logits_flat)
   # target_flat: (batch * max_len, 1)
   target_flat = target.view(-1, 1)
   # losses_flat: (batch * max_len, 1)
   losses_flat = -t.gather(log_probs_flat, dim=1, index=target_flat)
   # losses: (batch, max_len)
   losses = losses_flat.view(*target.size())
   # mask: (batch, max_len)
   mask = _sequence_mask(sequence_length=length, max_len=target.size(1))
   losses = losses * mask.float()
   loss = losses.sum() / length.float().sum()
   return loss

def runMinibatch(net, batcher, cuda=True, volatile=False, trainable=False):
   dat = batcher.next()
   #dat, human = batcher.next()
   x, y, mask = dat
   x = [var(e, volatile=volatile, cuda=cuda) if e is not None else e for e in x]
   y = [var(e, volatile=volatile, cuda=cuda) if e is not None else e for e in y]
   if mask is not None:
      mask = var(mask, volatile=volatile, cuda=cuda)

   if len(x) == 1:
      x = x[0]
   if len(y) == 1:
      y = y[0]

   a = net(x, trainable)#, human)
   return a, y, mask
     
def runData(net, opt, batcher, criterion=maskedCE, 
      trainable=False, verbose=False, cuda=True,
      gradClip=10.0, minContext=0, numPrints=10):
   iters = batcher.batches
   meanAcc  = CMA()
   meanLoss = CMA(n=1)

   errs = ErrorBreakdown()
   for i in range(iters):
      try:
         if verbose and i % int(iters/numPrints) == 0:
            sys.stdout.write('#')
            sys.stdout.flush()
      except: 
         pass

      #Always returns mask. None if unused
      a, y, mask = runMinibatch(net, batcher, trainable=trainable, cuda=cuda, volatile=not trainable)
      
      #Compute loss/acc with proper criterion/masking
      loss, acc = stats(criterion, a, y)
      progLabels = y[0].data.cpu().numpy()
      batch = progLabels.shape[0]
      ends = np.argmax(progLabels == 0, 1)
      ends = progLabels[np.arange(batch), ends-1]
      correct = (t.max(a[1], 1)[1] == y[1]).cpu().data.numpy().ravel()
      errs.update(ends, correct)
   
      random = False
      if random and trainable:
         opt.step()
         aa, yy, _ = runMinibatch(net, batcher, trainable=trainable, cuda=cuda, volatile=not trainable)
         lossOpt, _ = mathStats(criterion, aa, yy)
         if lossOpt.data[0] >= loss.data[0]:
            opt.reset()
      elif trainable:
         #Put this back above backward
         opt.zero_grad()
         loss.backward()
         if gradClip is not None:
            t.nn.utils.clip_grad_norm(net.parameters(), 
                  gradClip, norm_type=1)

         opt.step()

      #Accumulate average
      meanLoss.update([loss.data[0]])
      meanAcc.update(acc)
   errs.scores

   return meanLoss.cma, meanAcc.cma

def mathStats(criterion, a, y):
   l1 = nn.L1Loss()
   l2 = nn.MSELoss()
   aPred = a
   aAns  = y[1]
   loss = l1(aPred, aAns)# + l2(aPred, aAns)
   #aAns = t.round(y[1]).long()
   #loss = criterion(aPred, aAns)
   return loss, (0, loss.data[0])
 
def stats(criterion, a, y):
   #maskCriterion = nn.CrossEntropyLoss(ignore_index=0)
   pPred, aPred = a
   p, a = y
   batch, sLen, vocab = pPred.size()
   pPred = pPred.view(-1, vocab)

   loss = 0
   progAcc = 0
   test = p is None
   if not test:
      p = p.view(-1)
      progLoss = criterion(pPred, p)
      #progLoss = maskCriterion(pPred, p)
      loss += progLoss

      mask = p.data != 0
      numLabels = t.sum(mask)
      _, preds = t.max(pPred.data, 1)
      progAcc = t.sum(mask * (p.data == preds)) / numLabels
      if progAcc > 1.0:
         T()

   ansLoss  = criterion(aPred, a)
   loss += ansLoss 

   _, preds = t.max(aPred.data, 1)
   ansAcc = t.mean((a.data == preds).float())
   
   return loss, (progAcc, ansAcc)
