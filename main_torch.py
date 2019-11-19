from __future__ import division
from __future__ import print_function
import torch
import cv2
import editdistance
from DataLoader import DataLoader
from SamplePreprocessor import preprocess
import params
from classifier.cnn import  HTC,  DecoderType
class FilePaths:
	"filenames and paths to data"
	fnCharList = params.root_dir+'results/charList.txt'
	fnAccuracy = params.root_dir+'results/accuracy.txt'
	fnTrain = 'D:/projects/DATA/IAM/'
	fnInfer = params.root_dir+'data/test.png'
	fnCorpus = params.root_dir+'data/corpus.txt'


decoderType = DecoderType.BestPath
loader = DataLoader(FilePaths.fnTrain, 50, (128,32), 32)
loader.trainSet()
batch = loader.getNext()
model = HTC('backup',loader.charList, decoderType)
model.fit(loader)


#test
import torch.nn as nn
T,C,N,S,S_min = 50,20,16,30,10
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
ctc_loss = nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)
loss.backward()
#model.trainBatch(batch)

#	epoch = 0 # number of training epochs since start
#	bestCharErrorRate = float('inf') # best valdiation character error rate
#	noImprovementSince = 0 # number of epochs no improvement of character error rate occured
#	earlyStopping = 5 # stop training after this number of epochs without improvement
#	while True:
#		epoch += 1
#		print('Epoch:', epoch)
#
#		# train
#		print('Train NN')
#		loader.trainSet()
#		while loader.hasNext():
#			iterInfo = loader.getIteratorInfo()
#			batch = loader.getNext()
#			loss = model.trainBatch(batch)
#			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)
#
#		# validate
#		charErrorRate = validate(model, loader)
#		
#		# if best validation accuracy so far, save model parameters
#		if charErrorRate < bestCharErrorRate:
#			print('Character error rate improved, save model')
#			bestCharErrorRate = charErrorRate
#			noImprovementSince = 0
#			model.save()
#			open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
#		else:
#			print('Character error rate not improved')
#			noImprovementSince += 1
#
#		# stop training if no more improvement in the last x epochs
#		if noImprovementSince >= earlyStopping:
#			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
#			break



