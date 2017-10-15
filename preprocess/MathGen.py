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

class Op(Symbol):
   def __init__(self):
      self.validSuccessors = Num

class Plus(Op):
   def __init__(self):
      super().__init__()
      self.txt = '+'
      self.ind = 3

class Minus(Op):
   def __init__(self):
      super().__init__()
      self.txt = '-'
      self.ind = 4

class Mul(Op):
   def __init__(self):
      super().__init__()
      self.txt = '*'
      self.ind = 5

class Div(Op):
   def __init__(self):
      super().__init__()
      self.txt = '/'
      self.ind = 6

class Paren(Symbol):
   pass

class LParen(Paren):
   def __init__(self):
      super().__init__()
      self.validSuccessors = Num
      self.txt = '('
      self.ind = 7

class RParen(Paren):
   def __init__(self):
      super().__init__()
      self.validSuccessors = Op
      self.txt = ')'
      self.ind = 8

class Num(Symbol):
   def __init__(self, maxVal):
      super().__init__()
      self.validSuccessors = Op
      num = np.random.randint(0, maxVal)
      self.txt = str(num)
      self.ind = 9 + num


def randOp():
   ops = [Plus, Minus, Mul, Div]
   return np.random.choice(ops)

def genEqn(nums, maxLen):
   cur = Start()
   ret = [cur]
   for i in range(maxLen-2):
      nxt = cur.validSuccessors
      if type(nxt) == list:
         nxt = np.random.choice(nxt)

      if nxt == Num:
         nxt = nxt(nums)
      elif nxt == Op: 
         nxt = randOp()()
  
      ret += [nxt]
      cur = nxt
   ret += [End()]
   return ret

def eqnString(eqn):
   return ''.join([e.txt for e in eqn])

def eqnInds(eqn):
   return [e.ind for e in eqn]

def eqnAns(eqn):
   ind = 0
   val = eqn[0]
   eqn = eqn[1:]
   while len(eqn) > 1:
      nxt = eqn[:2]
      try:
         val = eval(str(val) + nxt)
      except:
         return None
      eqn = eqn[2:]
   return val

def genEqns(nums, maxLen, numSamples, split):
   with h5py.File('math.h5', 'a') as f:
      eqns = f.create_dataset('eqns' + split, (numSamples, maxLen), dtype='int32')
      ans = f.create_dataset('ans' + split, [numSamples])

      numEqns = 0
      while numEqns < numSamples:
         eqn = genEqn(nums, maxLen)
         a = eqnAns(eqnString(eqn))
         if a is None:
            continue

         eqns[numEqns] = eqnInds(eqn)
         ans[numEqns] = a
         numEqns += 1

      return eqns, ans

if __name__ == '__main__':
   maxLen = 19
   nums = 10
   for e in [(250000, 'train'), (10000, 'valid'), (25000, 'test')]:
      numSamples, split = e
      eqns, ans = genEqns(nums, maxLen, numSamples, split)
   print(ans)
   #print(eqnInds(eqn))   




