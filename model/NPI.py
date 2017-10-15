from pdb import set_trace as T
import torch as t
from torch import nn
from torch.nn import functional as F
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

class CNNFeats(nn.Module):
   def __init__(self, h, numClasses):
      super(CNNFeats, self).__init__()
      self.conv1  = utils.Conv2d(h, h, 3)
      self.conv2  = utils.Conv2d(h, h, 3)
      self.conv3  = utils.Conv2d(h, int(h/2), 3, padding='valid')
      self.fc1    = nn.Linear(int(h/2) * 5 * 5, 1024)
      self.pool   = nn.MaxPool2d(2)
      self.fc2    = nn.Linear(1024, numClasses)

   def forward(self, x) :
      inp = x
      x = F.relu(self.conv1(x))
      x = self.conv2(x)
      x += inp
      x = F.relu(x)
      x = self.pool(x)
      x = self.conv3(x)
      x = x.view(x.size()[0], -1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x

class UnaryModule(nn.Module):
   def __init__(self, h):
      super(UnaryModule, self).__init__()
      self.conv1 = utils.Conv2d(h, h, 3)
      self.conv2 = utils.Conv2d(h, h, 3)
      self.norm2  = nn.InstanceNorm2d(h)

   def forward(self, x):
      inp = x
      x = F.relu(self.conv1(x))
      x = self.conv2(x)
      x += inp
      x = self.norm2(F.relu(x))
      return x
      
class BinaryModule(nn.Module):
   def __init__(self, h):
      super(BinaryModule, self).__init__()
      self.conv1  = utils.Conv2d(2*h, h, 1)
      self.conv2  = utils.Conv2d(h, h, 3)
      self.conv3  = utils.Conv2d(h, h, 3)

   def forward(self, x1, x2):
      x = t.cat((x1, x2), 1)
      x = F.relu(self.conv1(x))
      res = x
      x = F.relu(self.conv2(x))
      x = self.conv3(x)
      x += res
      x = F.relu(x)
      return x

class BinFork(nn.Module):
   def __init__(self, h):
      super(BinFork, self).__init__()
      self.conv1  = utils.Conv2d(2*h, h, 1)
      self.conv2  = utils.Conv2d(h, h, 3)
      self.conv3  = utils.Conv2d(h, h, 3)

   def forward(self, x1, x2):
      x = t.cat((x1, x2), 1)
      x = F.relu(self.conv1(x))
      res = x
      x = F.relu(self.conv2(x))
      x = self.conv3(x)
      x += res
      return x

class BinForkLarge(nn.Module):
   def __init__(self, h):
      super(BinForkLarge, self).__init__()
      self.conv1  = utils.Conv2d(2*h, 4*h, 1)
      self.conv2  = utils.Conv2d(4*h, 4*h, 3)
      self.conv3  = utils.Conv2d(4*h, 4*h, 3)
      self.conv4  = utils.Conv2d(4*h, h, 1)

   def forward(self, x1, x2):
      x = t.cat((x1, x2), 1)
      x = F.relu(self.conv1(x))
      res = x
      x = F.relu(self.conv2(x))
      x = self.conv3(x)
      x += res
      x = F.relu(x)
      x = self.conv4(x)
      return x

class ForkModule(nn.Module):
   def __init__(self, h):
      super(ForkModule, self).__init__()
      self.conv1  = utils.Conv2d(2*h, 6*h, 1)
      self.conv2  = utils.Conv2d(6*h, 6*h, 3)
      self.conv3  = utils.Conv2d(6*h, 6*h, 3)
      self.conv4  = utils.Conv2d(6*h, h, 1)

   def forward(self, x1, x2):
      x = t.cat((x1, x2), 1)
      x = F.relu(self.conv1(x))
      res = x
      x = F.relu(self.conv2(x))
      x = self.conv3(x)
      x += res
      x = F.relu(x)
      x = self.conv4(x)
      return x

class NPI(nn.Module):
   def __init__(self,
         embedDim, hGen, hExe, qLen, qVocab,
         numUnary, numBinary, pLen, aVocab):
      super().__init__()
      padCells = 3
      unaries  = [UnaryModule(hExe) for i in range(numUnary)]
      binaries = [BinaryModule(hExe) for i in range(numBinary)]
      #fork = [BinForkLarge(hExe) for i in range(numBinary+1)]
      #self.fork = fork
      #binaries = [BinaryModule(hExe) for i in range(numBinary)]
      self.fork = ForkModule(hExe)
      prefix   = [Iden()]*3
      self.cells = nn.ModuleList(prefix + unaries + binaries)# + fork)
      self.cellDrop = nn.Dropout2d(p=0.0)

      self.controllerLSTM = RHNCell(hExe, hGen, depth=3)
      self.controllerCNN  = CNNFeats(hExe, hExe)
      self.classifier = CNNFeats(hExe, aVocab)
      self.CNN = ExecutionEngine.CNN(hExe)

      self.projController  = nn.Linear(hGen, numUnary + numBinary + padCells)

      self.embed = nn.Embedding(qVocab, embedDim)
      self.encoder = t.nn.LSTM(embedDim, hGen, 2, batch_first=True)

      self.pLen = pLen
      self.hGen = hGen
      self.numUnary = numUnary
      self.reward = utils.EDA()

   def forward(self, x, trainable=False, human=False):
      parallel = False
      q, prog, ans, imgs = x

      if True :#not human and trainable:
         prog = prog.data.cpu().numpy()
      progPreds = []

      #Encode the question
      x = self.embed(q)
      pred, state = self.encoder(x)
      hEnc, cEnc = state[0][-1], state[1][-1]

      batch, seq, hExe = pred.size()
      outImgs = [None]*batch

      #Initial image encoding 
      imgs = self.CNN(imgs)
      origImgs = imgs.clone()
      mem = [None]*batch
      pReinforce = []
      for i in range(self.pLen):
         #Controller
         imgEmbed = self.controllerCNN(imgs)
         state = self.controllerLSTM(imgEmbed, cEnc)

         #Predict prog cell
         pPred = self.projController(state)
         progPreds += [pPred]
 
         #Feed in prediction (val) or ground truth (train)
         if True:#not human and trainable:
            p = prog[:, i]
         else:
            _, p = t.max(pPred, 1)
            p = p.data.cpu().numpy()
         p = p.ravel().tolist()
         
         imgStack = []
         for j in range(batch):
            pj = p[j]
            if pj == 1: #fork #>42
               mem[j] = imgs[j:j+1]
               cell = self.fork #Binary
               #cell = self.cells[pj]
               imgStack += [cell(imgs[j:j+1], origImgs[j:j+1])]
               continue
            cell = self.cells[pj] 
            arg1 = imgs[j:j+1]
            if mem[j] is not None and pj > 32:
               #merge
               imgi = cell(mem[j], arg1)
            elif pj > 32:
               imgi = arg1
            else:
               imgi = cell(arg1)
            imgStack += [imgi]
         imgs = t.cat(imgStack, 0)

         '''
         #Execute prog cell
         cells = [self.cells[e] for e in p]
         imgStack = [cells[i](imgs[i:i+1]) for i in range(batch)]
         imgs = t.cat(imgStack, 0)
         '''
         #if trainable:
         #   imgs = self.cellDrop(imgs)

         #Lock in end token (or prevent crashing if never chosen)
         for j in range(batch):
            if outImgs[j] is None and (p[j] == 0 or i == self.pLen-1):
               outImgs[j] = imgs[j]
               
      #Final classifier
      progPreds = t.stack(progPreds, 1)
      imgs = t.stack(outImgs, 0)
      a = self.classifier(imgs)

      return progPreds, a
#Reinforce
'''
if human and trainable:
   if self.reward.eda is None:
      self.reward.eda = 0.0
   _, preds = t.max(a.data, 1)
   ansAcc = (ans.data == preds).float().view(-1, 1)
   accMean = t.mean(ansAcc)
   ansAcc -= self.reward.eda
   batch, seq, vocab = progPreds.size()
   reinforceNodes = []
   gradOuts = []
   for i in range(seq):
      progMult = progPreds[:, i].multinomial()
      progMult.reinforce(ansAcc)
      reinforceNodes += [progMult]
      gradOuts += [None]
   t.autograd.backward(reinforceNodes, gradOuts, retain_variables=True)
   self.reward.update(accMean)
'''

'''
def dynamicPool(x, p, cells, pIgnore=(0, 1)):
   grouped = {}
   out = {}
   sz = len(p)
   for i in range(sz):
      xi, pi = x[i], p[i]
      grouped.setdefault(pi, []).append([i, xi])
   for key, val in grouped.items():
      val = list(zip(*val))
      inds, xs = val
      if key not in pIgnore:
         try:
            xs = [e for e in cells[key](t.stack(xs, 0))]
         except:
            T()

      for i in range(len(inds)):
         ind, xi = inds[i], xs[i]
         out[ind] = xi
   return t.stack([e for e in out.values()], 0) 

         if parallel:
            imgs = dynamicPool(imgs, p, self.cells, pIgnore=(0, 1)) 
         else:
            cells = [self.cells[e] for e in p]
            imgStack = [cells[i](imgs[i:i+1]) for i in range(batch)]
            imgs = t.cat(imgStack, 0)

'''
