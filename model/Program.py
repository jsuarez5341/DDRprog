from pdb import set_trace as T
import numpy as np

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from itertools import groupby
import time

class Node():
   def __init__(self, prev):
      self.prev = prev 
      self.inpData = []

   def build(self, cellInd, mul, arity):
      self.next = [None] * arity
      self.arity = arity
      self.cellInd = cellInd
      self.mul = mul

class Program:
   def __init__(self, prog, mul, imgFeats, arities):
      self.prog = prog
      self.mul  = mul
      self.imgFeats = imgFeats
      self.arities = arities
      self.root = Node(None)

   def build(self, ind=0):
      self.buildInternal(self.root)

   def buildInternal(self, cur=None, count=0):
      if count >= len(self.prog):
         arity = 0
         ind = 0
         mul = 1.0
      else:
         ind = self.prog[count]
         mul = None
         if self.mul is not None:
            mul = self.mul[count:count+1]
         arity = self.arities[ind]

      cur.build(ind, mul, arity)

      if arity == 0:
         cur.inpData = [self.imgFeats]
      elif arity == 1:
         cur.next = [Node(cur)]
         count = self.buildInternal(cur.next[0], count+1)
      elif arity == 2:
         cur.next = [Node(cur), Node(cur)]
         count = self.buildInternal(cur.next[0], count+1)
         count = self.buildInternal(cur.next[1], count+1)

      return count

   def flat(self):
      return self.flatInternal(self.root, [])

   def flatInternal(self, cur, flattened):
      flattened += [cur.cellInd]
      for e in cur.next:
         self.flatInternal(e, flattened)

      return flattened

   def topologicalSort(self):
      return self.topInternal(self.root, [])

   def topInternal(self, cur, flattened):
      for e in cur.next:
         self.topInternal(e, flattened)

      flattened += [cur]
      return flattened

class HighArcESort:
   def __init__(self):
      self.out = {}

   def __call__(self, root):
      assert(not self.out) #Empty
      self.highArcESortInternal(root, 0)
      return self.out

   def highArcESortInternal(self, cur, rank):
      for nxt in cur.next:
         ret = self.highArcESortInternal(nxt, rank)
         rank = max(rank, ret)
      self.out[rank] = cur
      return rank+1

class FasterExecutioner:
   def __init__(self, progs, cells):#, upscale):
      self.cells = cells
      #self.upscale = upscale

      self.progs = progs
      self.roots = [p.root for p in progs]
      self.sortProgs()
      self.maxKey = max(list(self.progs.keys()))

   def sortProgs(self):
      progs = {}
      for prog in self.progs:
         prog = HighArcESort()(prog.root)
         for rank, nodeList in prog.items():
            progs.setdefault(rank, []).append(nodeList)
      self.progs = progs
   
   def execute(self):
      for s in range(self.maxKey+1):
         nodes = self.progs[s]
         groupedNodes =  {}
         for node in nodes:
            groupedNodes.setdefault(node.cellInd, []).append(node)

         for cellInd, nodes in groupedNodes.items():
            arity = nodes[0].arity
            cell = self.cells[cellInd]
            #upscale = self.upscale[cellInd]

            outData = [node.inpData[0] for node in nodes]
            if arity==1:
               arg = t.cat(outData, 0)
               outData = cell(arg)
               outData = [outData[i:i+1] for i in range(outData.size()[0])]
            elif arity==2:
               arg2 = t.cat(outData, 0)
               arg1 = t.cat([node.inpData[1] for node in nodes], 0)
               outData = cell(arg1, arg2) 
               outData = [outData[i:i+1] for i in range(outData.size()[0])]
        
            for node, outDat in zip(nodes, outData):
               if type(node.mul) != float:
                  #outDat = outDat * node.mul.expand_as(outDat)# * upscale(node.mumull)
                  if node.mul is not None:
                     outDat = outDat * node.mul

               if node.prev is None:
                  node.outData = outDat
               else:
                  node.prev.inpData += [outDat]

      outData = [root.outData for root in self.roots]
      return t.cat(outData, 0)


class FastExecutioner:
   def __init__(self, progs, cells):
      self.cells = cells

      self.progs = progs
      self.sortProgs()

   def sortProgs(self):
      for i in range(len(self.progs)):
         self.progs[i] = self.progs[i].topologicalSort()
   
   def execute(self):
      maxLen = max([len(e) for e in self.progs])
      for s in range(maxLen):
         nodes = []
         for i in range(len(self.progs)):
            prog = self.progs[i]
            if len(prog) <= s:
               continue
            nodes += [prog[s]]

         groupedNodes = {}
         for node in nodes:
            groupedNodes.setdefault(node.cellInd, []).append(node)

         for cellInd, nodes in groupedNodes.items():
            arity = nodes[0].arity
            cell = self.cells[cellInd]

            outData = [node.inpData[0] for node in nodes]
            if arity==1:
               arg = t.cat(outData, 0)
               outData = cell(arg)
               outData = t.split(outData, 1, 0)
            elif arity==2:
               arg1 = t.cat(outData, 0)
               arg2 = t.cat([node.inpData[1] for node in nodes], 0)
               outData = cell(arg1, arg2) 
               outData = t.split(outData, 1, 0)
            
            for node, outDat in zip(nodes, outData):
               if node.prev is None:
                  node.outData = outDat
               else:
                  node.prev.inpData += [outDat]

      outData = [prog[-1].outData for prog in self.progs]
      return t.cat(outData, 0)

class Executioner:
   def __init__(self, prog, cells):
      self.prog = prog
      self.cells = cells

   def execute(self):
      return self.executeInternal(self.prog.root)

   def executeInternal(self, cur):
      if cur.arity == 0:
         return cur.inpData[0]
      elif cur.arity == 1:
         args = [self.executeInternal(cur.next[0])]
      elif cur.arity == 2:
         arg1 = self.executeInternal(cur.next[0])
         arg2 = self.executeInternal(cur.next[1])
         args = [arg1, arg2]

      cell = self.cells[cur.cellInd]
      return cell(*args) * cur.mul


