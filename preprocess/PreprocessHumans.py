from pdb import set_trace as T
import numpy as np
import h5py
import json
from lib import utils
from lib import nlp
import unicodedata
import sys

tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))
def removePunctuation(text):
    return text.translate(tbl)

def expandVocab(vocab, questions, split):
   maxVal = np.max(list(vocab.values()))
   for e in questions:
      e = removePunctuation(e[split].lower()).split()
      for w  in e:
         if w not in vocab.keys():
            maxVal += 1
            vocab[w] = maxVal
   return vocab

def preprocessQuestions(questions, vocab, fAppend, maxLen=45):
   ret = []
   retImgIdx= []
   for e in questions:
      retImgIdx += [e['image_index']]
      e = removePunctuation(e['question'].lower()).split() + ['?']
      x = nlp.applyVocab(e, vocab).tolist()
      x += [0]*(maxLen - len(x))
      ret += [x]

   ret = np.asarray(ret)
   retImgIdx = np.asarray(retImgIdx)
   with h5py.File('data/preprocessed/clevr.h5', 'a') as f:
      data = f.create_dataset(fAppend+'HumanQuestions', data=ret)
      data = f.create_dataset(fAppend+'HumanImageIdx', data=retImgIdx)

def preprocessAnswers(answers, vocab, fAppend):
   ret = []
   for e in answers:
      e = e['answer'].lower().split()
      x = nlp.applyVocab(e, vocab).tolist()
      ret += [x]

   ret = np.asarray(ret)
   with h5py.File('data/preprocessed/clevr.h5', 'a') as f:
      data = f.create_dataset(fAppend+'HumanAnswers', data=ret)

def runTxt():
   splits = ['train', 'val']
   for split in splits:
      with open('data/CLEVR-Humans/CLEVR-Humans-'+split+'.json') as f:
         split = split[0].upper() + split[1:]
         dat = json.load(f)['questions']

         #Random permutation
         perm = np.random.permutation(np.arange(len(dat)))
         dat = (np.asarray(dat)[perm]).tolist()

         print('Preprocessing Questions...')
         questionF = 'data/vocab/QuestionVocab.txt'
         questionVocab, _ = nlp.buildVocab(questionF)
         questionVocab = expandVocab(questionVocab, dat, 'question')
         preprocessQuestions(dat, questionVocab, split)

         print('Preprocessing Answers...')
         answerF = 'data/vocab/AnswerVocab.txt'
         answerVocab, _ = nlp.buildVocab(answerF)
         answerVocab = expandVocab(answerVocab, dat, 'answer')
         preprocessAnswers(dat, answerVocab, split)

         print('Done')
