from pdb import set_trace as T
import h5py
import numpy as np

def run(linear=32):
   dat = h5py.File('data/preprocessed/clevr.h5', 'r')

   for split in ['Train', 'Val']:
      imgs    = dat[split + 'Imgs']
      programs  = dat[split + 'Programs']
      pMask     = dat[split + 'ProgramMask']

      questions = dat[split + 'Questions']
      answers   = dat[split + 'Answers']
      imgIdx    = dat[split + 'ImageIdx']


      batch = 1000
      q, a, idx, p, pm = [], [], [], [], []
      for i in range(len(programs)):
         if i % batch == 0:
            print(i)

         if np.max(programs[i]) <= linear:
            pi = programs[i]
            #prog = programs[i][:np.argmax(programs[i] == 0)]
            #pi *= 0
            #pi[:len(prog)] = prog

            q += [questions[i]]
            a += [answers[i]]
            idx += [imgIdx[i]]
            p += [pi]
            pm += [pMask[i]]
   
      q = np.stack(q, 0)
      a = np.stack(a, 0)
      idx = np.stack(idx, 0)
      p = np.stack(p, 0)
      pm = np.stack(pm, 0)
       
      with h5py.File('data/preprocessed/linearClevr.h5', 'a') as f:
         f.create_dataset(split+'Questions', data=q)
         f.create_dataset(split+'Answers', data=a)
         f.create_dataset(split+'ImageIdx', data=idx)
         f.create_dataset(split+'Programs', data=p)
         f.create_dataset(split+'ProgramMask', data=pm)

