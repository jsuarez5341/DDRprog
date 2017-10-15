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
   def __init__(self, val):
      super().__init__()
      self.val = 0.1*val
      self.ind = int(3 + 10*self.val)

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
   for tok in eqn:
      if issubclass(type(tok), Op):
         arg2 = stack.pop().val
         arg1 = stack.pop().val
         ret = tok(arg1, arg2)
         if ret is None:
            return None
         stack.append(eqnNum(ret))
      else:
         stack.append(tok)

   return stack.pop().val

def genEqns(nums, n, maxLen, numSamples, split):
   with h5py.File('exactRPN.h5', 'a') as f:
      eqns = f.create_dataset('eqns' + split, (numSamples, maxLen), dtype='int32')
      ans = f.create_dataset('ans' + split, [numSamples])

      numEqns = 0
      for i in range(10):
         for j in range(10):
            for op in [Plus(), Minus(), Mul(), Div()]:
               eqn = [Num(i), Num(j), op]
               a = eqnAns(eqn)

               if a is None:
                  continue

               eqn = [Start()] + eqn + [End()]
               eqnI = eqnInds(eqn)

               eqns[numEqns] = np.asarray(eqnI).astype(np.int32)
               ans[numEqns] = a
               numEqns += 1
      for i in range(10):
         eqn = [Num(0), Num(0), Plus()]
         a = eqnAns(eqn)
         eqn = [Start()] + eqn + [End()]
         eqnI = eqnInds(eqn)
         eqns[numEqns] = np.asarray(eqnI).astype(np.int32)
         ans[numEqns] = a
         numEqns += 1
 
      return eqns, ans

if __name__ == '__main__':
   n = 1
   maxLen = 2*n + 1 + 2 #start/end
   maxNum = 9
   for split in ['train', 'valid', 'test']:
      eqns, ans = genEqns(maxNum, n, maxLen, 400, split)
   print(ans)
   #print(eqnInds(eqn))   




