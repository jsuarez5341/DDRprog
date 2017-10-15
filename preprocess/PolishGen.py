from pdb import set_trace as T
import numpy as np
from enum import Enum
import h5py

class Symbol:
   pass

class Start(Symbol):
   def __init__(self):
      self.validSuccessors = [Num]
      self.txt = ''
      self.ind = 1

class End(Symbol):
   def __init__(self):
      self.txt = ''
      self.ind = 2

class Num(Symbol):
   def __init__(self, maxVal):
      super().__init__()
      self.val = 0.1*np.random.randint(0, maxVal+1)
      self.ind = 3 + int(10*self.val)

class Op(Symbol):
   pass

class Plus(Op):
   def __init__(self):
      super().__init__()
      self.txt = '+'
      self.ind = 13

   def __call__(self, arg1, arg2):
      return arg1 + arg2

class Minus(Op):
   def __init__(self):
      super().__init__()
      self.txt = '-'
      self.ind = 14

   def __call__(self, arg1, arg2):
      return arg1 - arg2


class Mul(Op):
   def __init__(self):
      super().__init__()
      self.txt = '*'
      self.ind = 15

   def __call__(self, arg1, arg2):
      return arg1 * arg2

class Div(Op):
   def __init__(self):
      super().__init__()
      self.txt = '/'
      self.ind = 16

   def __call__(self, arg1, arg2):
      if arg2 == 0:
         return None
      return arg1 / arg2

def randOp():
   ops = [Plus, Minus, Mul, Div]
   return np.random.choice(ops)

def genEqn(maxNum, n):
   nums = [Num(maxNum) for i in range(n+1)]
   ops  = [randOp()() for i in range(n)]
   return nums + ops

def eqnString(eqn):
   return ''.join([e.txt for e in eqn])

def eqnInds(eqn):
   return [e.ind for e in eqn]

def eqnAns(eqn):
   class eqnNum:
      def __init__(self, val):
         self.val = val

   stack = []
   retList = []
   for tok in eqn:
      if issubclass(type(tok), Op):
         arg2 = stack.pop().val
         arg1 = stack.pop().val
         ret = tok(arg1, arg2)
         if ret is None:
            return None
         retList += [ret]
         stack.append(eqnNum(ret))
      else:
         stack.append(tok)
   return retList

def genEqns(nums, n, maxLen, numSamples, split):
   with h5py.File('math.h5', 'a') as f:
      eqns = f.create_dataset('eqns' + split, (numSamples, maxLen), dtype='int32')
      ans = f.create_dataset('ans' + split, (numSamples, n))

      numEqns = 0
      while numEqns < numSamples:
         print(numEqns)
         eqn = genEqn(nums, n)
         a = eqnAns(eqn)
         eqn = [Start()] + eqn + [End()]
         if a is None or np.max(np.abs(a))>100:
            continue

         eqns[numEqns] = eqnInds(eqn)
         ans[numEqns] = a
         numEqns += 1

      return eqns, ans

if __name__ == '__main__':
   n = 5
   maxLen = 2*n + 1 + 2 #start/end
   maxNum = 9
   for e in [(100000, 'train'), (5000, 'valid'), (20000, 'test')]:
      numSamples, split = e
      eqns, ans = genEqns(maxNum, n, maxLen, numSamples, split)
   print(ans)
   #print(eqnInds(eqn))   




