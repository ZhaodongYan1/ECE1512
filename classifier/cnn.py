#!/usr/bin/python
# Build a classifier model with the VA features
# @author sjeblee@cs.toronto.edu

import keras 
import numpy as np 
import pandas as pd  
import params
import gc 
from DataLoader import DataLoader, Batch

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelTest:
    #batchSize = 100
    
    def __init__(self, output_directory, verbose=False):
        self.output_directory = output_directory
        self.imgSize = params.imgSize
        self.model = self.build_model()
        self.maxTextLen = 32
    def build_model(self):
        input_layer = keras.layers.Input((self.imgSize[0],self.imgSize[1]))
        conv1 = keras.layers.Conv2D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.normalization.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)
        
        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.normalization.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)
        
        conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
        conv3 = keras.layers.normalization.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)
        
        gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(conv3)
        
        output_layer = keras.layers.Dense(params.numLetters)(gap_layer)
        
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        model.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, 
			min_lr=0.0001)


        file_path = self.output_directory+'best_model.hdf5'
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)
        
        self.callbacks = [reduce_lr,model_checkpoint]
        
        return model 
    def fit(self,loader):
    	"train NN"
    	
    	epoch = 0 # number of training epochs since start
    	for epoch in range(params.EPOCHS):
    		print('Epoch:', epoch)
    
    		# train
    		print('Train NN')
    		loader.trainSet()
    		while loader.hasNext():
    			iterInfo = loader.getIteratorInfo()
    			batchX,batchY = loader.getNext()
    			hist = self.model.fit(batchX,batchY)
    			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', hist.history['loss'])
    			gc.collect()                 


class CNN_ELMO(nn.Module):
    def __init__(self, output_directory):
          super(CNN_ELMO, self).__init__()

          self.output_directory = output_directory
          self.imgSize = params.imgSize
          self.model = self.build_model()
          self.maxTextLen = 32
          
          self.kernelVals = [5,5,3,3,3]
          self.featureVals = [1,32,64,128,128,256]
          self.poolVals = [(2,2),(2,2),(1,2),(1,2),(1,2)]
          self.strideVals = self.poolVals
    ''' Input is a list of strings
    '''
    def forward(self, x):
        
        #Initially x is of shape (batch_size,32,128)
         conv1 = nn.Conv2d(self.featureVals[0], self.featureVals[1], self.kernelVals[0])
         x = F.relu(conv1(x))
         x = F.max_pool1d(self.poolVals[0],stride=self.strideVals[0])

         conv2 = nn.Conv2d(self.featureVals[1], self.featureVals[2], self.kernelVals[1])
         x = F.relu(conv2(x))
         x = F.max_pool1d(self.poolVals[1],stride=self.strideVals[1])
         
         
         conv3 = nn.Conv2d(self.featureVals[2], self.featureVals[3], self.kernelVals[0])
         x = F.relu(conv3(x))
         x = F.max_pool1d(self.poolVals[2],stride=self.strideVals[2])
         
         conv4 = nn.Conv2d(self.featureVals[3], self.featureVals[4], self.kernelVals[0])
         x = F.relu(conv4(x))
         x = F.max_pool1d(self.poolVals[3],stride=self.strideVals[3])
         
         conv5 = nn.Conv2d(self.featureVals[4], self.featureVals[5], self.kernelVals[0])
         x = F.relu(conv5(x))
         x = F.max_pool1d(self.poolVals[4],stride=self.strideVals[4])
         
         #shape of x is (batch_size,32,256)
         
         
         
         
         x = self.dropout(x)  # (N, len(Ks)*Co)
         return x

    ''' Creates and trains a cnn neural network model. 
        X: a list of training data (string) 
        Y: a list of training labels (int)
        WARNING: Currently you can't use the encoding layer and use_prev_labels at the same time
    '''
    def fit(self, X, Y,batch_size=16,num_epochs=12,learning_rate=0.001,dropout=0.0):
#        start = time.time()
        # Parameters
        
        hidden_size = self.hidden_size
        dropout = self.dropout

        print("batch_size:", str(batch_size), "dropout:", str(dropout), "epochs:", str(num_epochs))
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        if use_cuda:
            self = self.to(cuda)
        Y = Y.astype('int') 
        X_len = X.shape[0]
        steps = 0

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        st = time.time()
        steps = 0
        for epoch in range(num_epochs):
            print("epoch", str(epoch))
            i = 0
            numpy.random.seed(seed=1)
            permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
            Xiter = X[permutation]
            Yiter = Y[permutation]
    
            while True:
                 if i+batch_size < X_len:
                     batchX = Xiter[i:i+batch_size]
                     batchY = Yiter[i:i+batch_size]   
                 else:
                     batchX = Xiter[i:]
                     batchY = Yiter[i:] 
    #                 print('-------------%d----------------------'%i)
                 batchX = torch.tensor(batchX)
                 if use_cuda:
                     character_ids = batchX.to(cuda).long()
                 else:
                     character_ids = batchX
                 character_ids.contiguous()
                 Xtensor = self.elmo(character_ids)
                 Xtensor = Xtensor['elmo_representations'][0].float()
                 Ytensor = torch.from_numpy(batchY).long()
                 del batchX
                 del batchY
                 if use_cuda:
                     Xtensor = Xtensor.cuda()
                     Ytensor = Ytensor.cuda()
                 feature = Variable(Xtensor)
                 target = Variable(Ytensor)
                 del Xtensor
                 del Ytensor
                 i = i+batch_size
                 
                 optimizer.zero_grad() 
#                 print(feature.size())
                 output = self(feature)
                 loss = F.cross_entropy(output, torch.max(target,1)[1])
                 loss.backward()
                 optimizer.step()
    
                 steps += 1
                 if i >= X_len:
                     break
            ct = time.time() - st
            unit = "s"
            if ct > 60:
                ct = ct/60
                unit = "m"
            emacs("time so far: ", str(ct), unit)

    def predict(self, testX,testX2,batch_size=16, keep_list=True, return_encodings=False,num_word=200):
        # Test the Model
        i = 0
        if debug: print("testX len:", str(len(testX)))
        print("testing...")
        stime = time.time()
        testX = torch.tensor(testX).long()


        y_pred = []
        logsoftmax = nn.LogSoftmax(dim=1)
        i = 0
    #    print(testX.size(0),'size')
        while True:
            if i+batch_size<testX.size(0):
    #        print(i)
                batchX = testX[i:i+batch_size]
            else: 
                batchX = testX[i:]
            if use_cuda:
                character_ids = batchX.to(cuda)
            character_ids.contiguous()
            Xtensor = self.elmo(character_ids)
            Xtensor = Xtensor['elmo_representations'][0].float()
            icd_var = self(Variable(Xtensor))
            
            icd_vec = logsoftmax(icd_var)
            for j in range(icd_vec.size(0)):
    #            print('icd_vec',icd_vec[i,:].size())
                icd_code = torch.max(icd_vec[j:j+1,:], 1)[1].data[0]
                icd_code = icd_code.item()
                y_pred.append(icd_code)
            i = i+batch_size
            if i >= testX.size(0):
                break
        print("testX shape: " + str(testX.shape))

        etime = time.time()
        print("testing took " + str(etime - stime) + " s")
        return y_pred