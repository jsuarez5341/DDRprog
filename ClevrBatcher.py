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

class ClevrBatcher():
   def __init__(self, batchSize, split, maxSamples=None, rand=True, human=False, linear=False, stack=False):

      dat = h5py.File('data/preprocessed/clevr.h5', 'r')
      self.imgs    = dat[split + 'Imgs']
      if linear:
         dat = h5py.File('data/preprocessed/linearClevr.h5', 'r')
      elif stack:
         dat = h5py.File('data/preprocessed/stackClevr.h5', 'r')

      if human:
         split += 'Human'
         self.programs = None
         self.pMask = None
      else:
         self.programs  = dat[split + 'Programs']
         self.pMask     = dat[split + 'ProgramMask']

      self.questions = dat[split + 'Questions']
      self.answers   = dat[split + 'Answers']
      self.imgIdx    = dat[split + 'ImageIdx']

      self.human = human
      self.batchSize = batchSize
      if maxSamples is not None: 
         self.m = maxSamples
      else:
         self.m = len(self.questions)//batchSize*batchSize
      self.batches = self.m // batchSize
      self.pos = 0
 
   def next(self):
      batchSize = self.batchSize
      if (self.pos + batchSize) > self.m:
         self.pos = 0

      #Hack to fix stupid h5py indexing bug
      imgIdx    = self.imgIdx[self.pos:self.pos+batchSize]
      uniqueIdx = np.unique(imgIdx).tolist()
      mapTo = np.arange(len(uniqueIdx)).tolist()
      mapDict = dict(zip(uniqueIdx, mapTo))
      relIdx = [mapDict[x] for x in imgIdx]

      imgs      = self.imgs[np.unique(imgIdx).tolist()][relIdx] #Hack to fix h5py unique indexing bug
      questions = self.questions[self.pos:self.pos+batchSize]
      answers   = self.answers[self.pos:self.pos+batchSize]

      if not self.human:
         programs  = self.programs[self.pos:self.pos+batchSize]
         pMask     = self.pMask[self.pos:self.pos+batchSize]
      else:
         programs = None
         pMask = None

      self.pos += batchSize
      return [questions, imgs, imgIdx], [programs, answers], [pMask]

   def vis(self, dat):
      print('WARNING: only supports train split. This is for sanity checking only')
      x, y, m = dat
      q, img, idx = x
      p, a = y
      mask = m 

      _, qVocab = nlp.buildVocab('data/vocab/QuestionVocab.txt')
      _, pVocab = nlp.buildVocab('data/vocab/ProgramVocab.txt')
      _, aVocab = nlp.buildVocab('data/vocab/AnswerVocab.txt')

      def apply(ary, vocab):
         ret = []
         for e in ary:
            if e in vocab.keys():
               ret += [vocab[e]]
         return ret
             

      for i in range(len(q)):
         qi, imgi, idxi, pi, ai = (
               q[i], img[i], idx[i], p[i], a[i])

         idxi = '0'*(6-len(str(idxi))) + str(idxi)
         imgi = imread('data/clevr/images/train/CLEVR_train_'+idxi+'.png')
         qi = ' '.join(apply(qi, qVocab))
         pi = ' '.join(apply(pi, pVocab))
         ai = ' '.join(apply(ai, aVocab))
         plt.imshow(imgi)
         plt.title(qi)
         plt.xlabel(pi + '  :  ' + ai)
         plt.show()




      
      
