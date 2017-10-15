from pdb import set_trace as T 
import sys
import time
from itertools import chain
import numpy as np
import torch
import torch as t
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable 
from torch.autograd import StochasticFunction

from lib import utils
from lib.RandomOptim import RandomOptim
from ClevrBatcher import ClevrBatcher
from MathBatcher import MathBatcher
from model.ExecutionEngine import ExecutionEngine
from model.ProgramGenerator import ProgramGenerator
from model.NPI import NPI
from model.MathNPI import NPILSTM
from model.MathNPI import NPICells

#Load PTB 
def dataBatcher(batchSz, maxSamples, human, stack):
   print('Loading Data...')

   trainBatcher = ClevrBatcher(batchSz, 'Train', maxSamples=maxSamples, human=human, linear=linear, stack=stack) 
   validBatcher = ClevrBatcher(batchSz, 'Val', maxSamples=maxSamples, human=human, linear=linear, stack=stack)
   print('Data Loaded.')

   return trainBatcher, validBatcher

def sampleBatcher(batchSz, maxSamples, humanProb=1.0):
   class Batcher:
      def __init__(self, clevr, human, humanProb=0.0):
         self.clevr = clevr
         self.human = human
         self.batches = self.clevr.batches
         self.humanProb = humanProb

      def next(self):
         if np.random.rand() < self.humanProb:
            return self.human.next(), True
         return self.clevr.next(), False

   trainClevr = ClevrBatcher(batchSz, 'Train', maxSamples=maxSamples, human=False, linear=linear) 
   validClevr = ClevrBatcher(batchSz, 'Val', maxSamples=maxSamples, human=False, linear=linear)

   trainHuman = ClevrBatcher(batchSz, 'Train', maxSamples=maxSamples, human=True, linear=linear) 
   validHuman = ClevrBatcher(batchSz, 'Val', maxSamples=maxSamples, human=True, linear=linear)

   trainClevr = NPIBatcher(trainClevr)
   validClevr = NPIBatcher(validClevr)
   trainHuman = NPIBatcher(trainHuman)
   validHuman = NPIBatcher(validHuman)

   trainBatcher = Batcher(trainClevr, trainHuman, humanProb)
   validBatcher = Batcher(validClevr, validHuman, 0.0)
   print('Data Loaded.')

   return trainBatcher, validBatcher


def mathBatcher(batchSz, maxSamples):
   print('Loading Data...')
   trainBatcher = MathBatcher(batchSz, 'train', maxSamples=maxSamples)
   validBatcher = MathBatcher(batchSz, 'valid', maxSamples=maxSamples)
   print('Data Loaded.')

   return trainBatcher, validBatcher

class NPIBatcher():
   def __init__(self, batcher):
      self.batcher = batcher
      self.batches = batcher.batches

   def next(self):
      x, y, mask = self.batcher.next()
      q, img, imgIdx = x
      p, ans = y
      pMask = mask[0]
      return [q, p, ans[:, 0], img], [p, ans[:, 0]], None

def train():
   epoch = -1
   while epoch < maxEpochs:
      epoch += 1

      start = time.time()
      trainLoss, trainAcc = utils.runData(net, opt, trainBatcher, 
            criterion, trainable=True, verbose=True, cuda=cuda)
      validLoss, validAcc = utils.runData(net, opt, validBatcher,
            criterion, trainable=False, verbose=False, cuda=cuda)
      trainEpoch = time.time() - start

      print('\nEpoch: ', epoch, ', Time: ', trainEpoch)
      print('| Train Perp: ', float(str(trainLoss[0])[:6]), 
            ', Train Acc: ', [float(str(e)[:6]) for e in trainAcc])
      print('| Valid Perp: ', float(str(validLoss[0])[:6]), 
            ', Valid Acc: ', [float(str(e)[:6]) for e in validAcc])

      saver.update(net, trainLoss, trainAcc, validLoss, validAcc)

def test():
   start = time.time()
   validLoss, validAcc = utils.runData(net, opt, validBatcher,
         criterion, trainable=False, verbose=True, cuda=cuda)

   print('| Valid Perp: ', validLoss, 
         ', Valid Acc: ', validAcc)
   print('Time: ', time.time() - start)


if __name__ == '__main__':
   load = False
   validate = False
   human = False
   multibatch = False
   linear = False
   stack = False
   cuda = True#All the cudas
   root='saves/' + sys.argv[1] + '/'
   saver = utils.SaveManager(root)
   maxSamples = None
   model = 'math'
   
   #Hyperparams
   embedDim = 300
   eta = 1e-4

   #Params
   humanProb = 0.00
   maxEpochs = 2000000
   batchSz = 100
   hGen = 128
   hExe = 64
   qLen = 45
   qVocab = 963 #96
   pVocab = 41
   pLen = qLen
   numUnary = 30
   numBinary = 9
   numClasses = 29

   if model == 'npi':
      if not multibatch:
         trainBatcher, validBatcher = dataBatcher(batchSz, 
               maxSamples, human=human, stack=stack)
         trainBatcher = NPIBatcher(trainBatcher)
         validBatcher = NPIBatcher(validBatcher)
      else:
         T()
         #trainBatcher, validBatcher = sampleBatcher(batchSz, maxSamples, humanProb)
      
 
      net = NPI( embedDim, hGen, hExe, qLen, qVocab,
            numUnary, numBinary, pLen, numClasses)
      criterion = nn.CrossEntropyLoss()
   if model == 'math':
      #Hyperparams
      mm = 32
      embedDim = mm
      eta = 5e-3

      #Params
      h = mm
      pVocab = 17
      numUnary = 12
      numBinary = 4
 
      trainBatcher, validBatcher = mathBatcher(batchSz, maxSamples)
      net = NPICells(embedDim, h, pVocab, numUnary, numBinary)
      #net = NPILSTM(embedDim, h, qLen, qVocab, numClasses)
      criterion = nn.CrossEntropyLoss()

   #utils.initWeights(net, scheme='normal')
   if load:
      saver.load(net)
   
   #net = t.nn.DataParallel(net, device_ids=[0, 1, 2, 3])

   if cuda:
      net.cuda()

   params = net.parameters()
   opt = t.optim.Adam(params, lr=eta)
   #opt = RandomOptim(net, lr=eta)
   utils.modelSize(net)
   
   if not validate:
      train()
   else:
      test()
