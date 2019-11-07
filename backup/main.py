import numpy as np
import pandas as pd 
import time
import logging
import params
from DataLoader import DataLoader
from classifier import cnn

import gc
from keras.models import load_model
from sklearn.metrics import mean_squared_error

def main():
    "main function"

    if params.train:
        model = cnn.ModelTest(params.output_directory,verbose=True)
        
        loader = DataLoader(params.data_dir, params.batchSize, params.imgSize, params.maxTextLen)
        model.fit(loader)
#        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
#        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))
        
        
        
        
#	# train or validate on IAM dataset	
#	if args.train or args.validate:
#		# load training data, create TF model
#		loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
#
#		# save characters of model for inference mode
#		open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
#		
#		# save words contained in dataset into file
#		open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))
#
#		# execute training or validation
#		if args.train:
#			model = Model(loader.charList, decoderType)
#			train(model, loader)
#		elif args.validate:
#			model = Model(loader.charList, decoderType, mustRestore=True)
#			validate(model, loader)
#
#	# infer text on test image
#	else:
#		print(open(FilePaths.fnAccuracy).read())
#		model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
#		infer(model, FilePaths.fnInfer)


if __name__ == '__main__':
	main()