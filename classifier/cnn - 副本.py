#!/usr/bin/python
# Build a classifier model with the VA features
# @author sjeblee@cs.toronto.edu

import keras 
import numpy as np 
import pandas as pd  
import params
import gc 
from DataLoader import DataLoader, Batch
import editdistance
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict 
use_cuda = torch.cuda.is_available()
if use_cuda:
    cuda = torch.device("cuda:0")
class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2
    
class HTC(nn.Module):
    batchSize = 50
    imgSize = (128,32)
    maxTextLen = 32
    def __init__(self, output_directory,charList,decoderType=DecoderType.BestPath):
          super(HTC, self).__init__()
          self. charList = charList
          self.output_directory = output_directory
          self.decoderType = decoderType
          self.imgSize = params.imgSize
          self.maxTextLen = 32
          
          self.kernelVals = [5,5,3,3,3]
          self.featureVals = [1,32,64,128,128,256]
          self.poolVals = [(2,2),(2,2),(1,2),(1,2),(1,2)]
          self.strideVals = [(2,2),(2,2),(1,2),(1,2),(1,2)]
          self.lstm = nn.LSTM(256,40, 2, batch_first=True,bidirectional=True)  
          self.batchesTrained = 0
           
    ''' Input is a list of strings
    '''
    def forward(self, x):
         # Initially x is numpy array of shape (batch,height,Width)
             
         x = torch.unsqueeze(x,1)
        # Shape of x: (batch,number of channels, H=128,W=32)
        
         # formula for conv2d: Hout = (Hin+2*pad[0]-1*kernel_size[0]-1)/stride[0]+1; same for w
         conv1 = nn.Conv2d(self.featureVals[0], self.featureVals[1], self.kernelVals[0],padding=(2,2))
         x = F.relu(conv1(x))
         x = F.max_pool2d(x,self.poolVals[0],stride=self.strideVals[0])

         conv2 = nn.Conv2d(self.featureVals[1], self.featureVals[2], self.kernelVals[1],padding=(2,2))
         x = F.relu(conv2(x))
         x = F.max_pool2d(x,self.poolVals[1],stride=self.strideVals[1])
         
         conv3 = nn.Conv2d(self.featureVals[2], self.featureVals[3], self.kernelVals[2],padding=(1,1))
         x = F.relu(conv3(x))
         x = F.max_pool2d(x,self.poolVals[2],stride=self.strideVals[2])
         
         conv4 = nn.Conv2d(self.featureVals[3], self.featureVals[4], self.kernelVals[3],padding=(1,1))
         x = F.relu(conv4(x))
         x = F.max_pool2d(x,self.poolVals[3],stride=self.strideVals[3])

         conv5 = nn.Conv2d(self.featureVals[4], self.featureVals[5], self.kernelVals[4],padding=(1,1))
         x = F.relu(conv5(x))
         x = F.max_pool2d(x,self.poolVals[4],stride=self.strideVals[4])
         
         # Shape of x: (batch,256,32,1)
         
         
         x = torch.squeeze(x,-1)   # Shape: batch,256,32
         x = torch.transpose(x,1,2)  # Shape: (batch,sequence=32,feature=256)
         
#         self.lstmSeq =nn.Sequential(OrderedDict([
#          ('LSTM1', nn.LSTM(256, 256, 1)),
#          ('LSTM1', nn.LSTM(256, 40, 1)),
#          ])) 
         hidden = torch.zeros(2,50,40)
         h0 = torch.randn(4,50, 40)
         c0 = torch.randn(4,50, 40)
         x2,hidden=self.lstm(x,(h0,c0))            
         # Shape of x2: batch,32,80
         x2 = x2.transpose(0,1).contiguous()
         # Shape of x2: 32,batch,80
         return x2

    ''' Creates and trains a cnn neural network model. 
        X: a list of training data (string) 
        Y: a list of training labels (int)
        WARNING: Currently you can't use the encoding layer and use_prev_labels at the same time
    '''
#    def toSparse(self, texts):
#        '''
#        put ground truth texts into sparse tensor for ctc_loss
#        indices->(text index, character index in its text)
#        values->index of the character in charList
#        shape->[number of texts,max number of characters among texts]
#        '''
#        indices = []
#        values = []
#        shape = [len(texts), 0] # last entry must be max(labelList[i])
#
#		# go over all texts
#        for (batchElement, text) in enumerate(texts):
#			# convert to string of label (i.e. class-ids)
#            labelStr = [self.charList.index(c) for c in text]
#			# sparse tensor must have size of max. label-string
#            if len(labelStr) > shape[1]:
#                shape[1] = len(labelStr)
#			# put each label into sparse tensor
#
#            for (i, label) in enumerate(labelStr):
#                indices.append([batchElement, i])
#                values.append(label)
#
#        return (indices, values, shape)
    def toSparse(self, texts):
        '''
        put ground truth texts into sparse tensor for ctc_loss
        indices->(text index, character index in its text)
        values->index of the character in charList
        shape->[number of texts,max number of characters among texts]
        '''
        values = []
        lengths = []
		# go over all texts
        for (batchElement, text) in enumerate(texts):
            labelStr = [self.charList.index(c) for c in text]
            if len(labelStr) > self.maxTextLen:
                labelStr = labelStr[:self.maxTextLen]
            lengths.append(len(labelStr))
            labelStr += [0]*(self.maxTextLen-len(labelStr))
            values.append(labelStr)
        return values,lengths

    def trainBatch(self,batch):
        numBatchElements = len(batch.imgs)
        sparse,lengths = self.toSparse(batch.gtTexts)
        learning_rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
        optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        x = torch.tensor(batch.imgs).float()
        output = self.forward(x)
        input_lengths = torch.full(size=(numBatchElements,), fill_value=output.size(0), dtype=torch.long)
        target_lengths = torch.tensor(lengths).long()
        ctc_loss = nn.CTCLoss()
        target = torch.tensor(sparse).long()
#        print(output[0],target[0])
        loss = ctc_loss(output,target,input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        self.batchesTrained += 1
        return loss
    def fit(self, loader):
#        start = time.time()
        # Parameters
    	"train NN"
    	if use_cuda:
            self = self.to(cuda)
    	epoch = 0 # number of training epochs since start
    	bestCharErrorRate = float('inf') # best valdiation character error rate
    	noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    	earlyStopping = 5 # stop training after this number of epochs without improvement
    	while True:
    		epoch += 1
    		print('Epoch:', epoch)
    
    		# train
    		print('Train NN')
    		loader.trainSet()
    		while loader.hasNext():
    			iterInfo = loader.getIteratorInfo()
    			batch = loader.getNext()
    			loss = self.trainBatch(batch)
    			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)
    
    		# validate
    		charErrorRate = self.validate(self, loader)
#    		
    		# if best validation accuracy so far, save model parameters
    		if charErrorRate < bestCharErrorRate:
    			print('Character error rate improved, save model')
    			bestCharErrorRate = charErrorRate
    			noImprovementSince = 0
    			self.save()
#    			open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
    		else:
    			print('Character error rate not improved')
    			noImprovementSince += 1
    
    		# stop training if no more improvement in the last x epochs
    		if noImprovementSince >= earlyStopping:
    			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
    			break
    	return
    def decoderOutputToText(self, ctcOutput, batchSize):
        "extract texts from output of CTC decoder"
		
		# contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(batchSize)]

		# word beam search: label strings terminated by blank
        if self.decoderType == DecoderType.WordBeamSearch:
            blank=len(self.charList)
            for b in range(batchSize):
                for label in ctcOutput[b]:
                    if label==blank:
                        break
                    encodedLabelStrs[b].append(label)

		# TF decoders: label strings are contained in sparse tensor
        else:
			# ctc returns tuple, first element is SparseTensor 
            decoded=ctcOutput[0][0] 

			# go over all indices and save mapping: batch -> values
#            idxDict = { b : [] for b in range(batchSize) }
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0] # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)

		# map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]
    def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
        "feed a batch into the NN to recognize the texts"
		
		# decode, optionally save RNN output
        numBatchElements = len(batch.imgs)
        ctcOutput = self.forward(batch.imgs)
        texts = self.decoderOutputToText(ctcOutput, numBatchElements)
		
#		# feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
#        if calcProbability:
#            sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
#            ctcInput = evalRes[1]
#            evalList = self.lossPerElement
#            feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
#            lossVals = self.sess.run(evalList, feedDict)
#            probs = np.exp(-lossVals)
#
#		# dump the output of the NN to CSV file(s)
#        if self.dump:
#            self.dumpNNOutput(evalRes[1])

        return (texts, probs)
    def validate(self,loader):
    	"validate NN"
    	print('Validate NN')
    	loader.validationSet()
    	numCharErr = 0
    	numCharTotal = 0
    	numWordOK = 0
    	numWordTotal = 0
    	while loader.hasNext():
    		iterInfo = loader.getIteratorInfo()
    		print('Batch:', iterInfo[0],'/', iterInfo[1])
    		batch = loader.getNext()
    		(recognized, _) = self.inferBatch(batch)
    		
    		print('Ground truth -> Recognized')	
    		for i in range(len(recognized)):
    			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
    			numWordTotal += 1
    			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
    			numCharErr += dist
    			numCharTotal += len(batch.gtTexts[i])
    			print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
    	
    	# print validation result
    	charErrorRate = numCharErr / numCharTotal
    	wordAccuracy = numWordOK / numWordTotal
    	print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
    	return charErrorRate
