batchSize = 100
imgSize = (128,32)
maxTextLen = 32
initialLearningRate = 0.01
momentum = 0.8


dump = False

preprocess = True
EPOCHS = 10
numLetters = 26
maxTextLen = 32
batchSize = 100

root_dir = 'D:/projects/Git/ECE1512/'
fnTrain = 'D:/projects/DATA/IAM/'
outputDir = 'results/'

decoderType = 'BestPath'   # Select one of ['BestPath','BeamSearch','WordBeamSearch']





fnCharList = root_dir+'results/charList.txt'
fnAccuracy = root_dir+'results/accuracy.txt'
fnInfer = root_dir+'data/test.png'
fnCorpus = root_dir+'data/corpus.txt'