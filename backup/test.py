from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
from SamplePreprocessor import preprocess
from scipy import misc
import cv2
from tqdm import tqdm
class Sample:
	"sample from the dataset"
	def __init__(self, gtText, filePath):
		self.gtText = gtText
		self.filePath = filePath


class Batch:
	"batch containing images and ground truth texts"
	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts

def truncateLabel( text, maxTextLen):
	# ctc_loss can't compute loss if it cannot find a mapping between text label and input 
	# labels. Repeat letters cost double because of the blank symbol needing to be inserted.
	# If a too-long label is provided, ctc_loss returns an infinite gradient
	cost = 0
	for i in range(len(text)):
		if i != 0 and text[i] == text[i-1]:
			cost += 2
		else:
			cost += 1
		if cost > maxTextLen:
			return text[:i]
	return text
import params
"loader for dataset at given location, preprocess images and text according to parameters"
filePath = params.data_dir
samples = []
currIdx = 0
batchSize = params.batchSize
imgSize = params.imgSize
assert filePath[-1]=='/'

f=open(filePath+'words.txt','r')
chars = set()
bad_samples = []
bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
for line in f:
	# ignore comment line
	if not line or line[0]=='#':
		continue
			
	lineSplit = line.strip().split(' ')
	assert len(lineSplit) >= 9
	
	# filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
	fileNameSplit = lineSplit[0].split('-')
	fileName = filePath + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

	# GT text are columns starting at 9
	gtText = truncateLabel(' '.join(lineSplit[8:]), params.maxTextLen)
	chars = chars.union(set(list(gtText)))

	# check if image is not empty
	if not os.path.getsize(fileName):
		bad_samples.append(lineSplit[0] + '.png')
		continue

	# put sample into list
	samples.append(Sample(gtText, fileName))

		# some images in the IAM dataset are known to be damaged, don't show warning for them
if set(bad_samples) != set(bad_samples_reference):
	print("Warning, damaged images found:", bad_samples)
	print("Damaged images expected:", bad_samples_reference)

# split into training and validation set: 95% - 5%
splitIdx = int(0.95 * len(samples))
trainSamples = samples[:splitIdx]
validationSamples = samples[splitIdx:]

# put words into lists
trainWords = [x.gtText for x in trainSamples]
validationWords = [x.gtText for x in validationSamples]

# number of randomly chosen samples per epoch for training 
numTrainSamplesPerEpoch = 25000 
		
# start with train set


# list of all chars in dataset
charList = sorted(list(chars))



