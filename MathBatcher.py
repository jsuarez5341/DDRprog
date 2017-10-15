import numpy as np
from pdb import set_trace as T
import os
import json
import h5py
import time
from matplotlib import pyplot as plt
from scipy.misc import imread

from lib import utils
from lib import nlp
from model.Tree import BTree

class MathBatcher():
   def __init__(self, batchSize, split, maxSamples=None, rand=True):

      dat = h5py.File('data/preprocessed/math5.h5', 'r')

      self.eqns  = dat['eqns'+split]
      self.ans = dat['ans'+split]

      self.batchSize = batchSize
      if maxSamples is not None: 
         self.m = maxSamples
      else:
         self.m = len(self.eqns)//batchSize*batchSize
      self.batches = self.m // batchSize
      self.pos = 0
 
   def next(self):
      batchSize = self.batchSize
      if (self.pos + batchSize) > self.m:
         self.pos = 0

      eqns = self.eqns[self.pos:self.pos+batchSize]
      ans  = self.ans[self.pos:self.pos+batchSize]

      self.pos += batchSize
      return [eqns, ans], [eqns, ans], None

