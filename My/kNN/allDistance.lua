-- This script calculate all distance matrixes between train features and test features in their corresponding config files under the given folder.
--require('mobdebug').start()
require 'cutorch'
dofile('/home/ljm/torch/My/kNN/distance.lua')
dofile('/home/ljm/torch/My/get_train_test_config.lua')
dofile('/home/ljm/torch/My/kNN/setPathes.lua')
dofile('/home/ljm/torch/My/kNN/kNN.lua')

-- define some parameters
testMaxSize = 24000
trainMaxSize = 22000
featureDimensionSize = 4608
-- create two feature matrixes and a distance matrix
trainFeatureMatrix = torch.CudaTensor(trainMaxSize, featureDimensionSize)
testFeatureMatrix = torch.CudaTensor(testMaxSize, featureDimensionSize)
distanceMatrix = torch.CudaTensor(trainMaxSize, testMaxSize)
-- get train and test config names
trainConfigNames, testConfigNames = getTTConfigNames(configRoot)

for i = 1, #trainConfigNames do
	print('train: ' .. trainConfigNames[i])
	print('test: ' .. testConfigNames[i])
	trainLabels, testLabels, testNames = distance(trainConfigNames[i], testConfigNames[i], trainFeatureMatrix, testFeatureMatrix, distanceMatrix)
	distanceM = torch.Tensor(#trainLabels,#testLabels)
	distanceM:copy(distanceMatrix[{ {1,#trainLabels}, {1,#testLabels} }])
	kNN(distanceM, trainLabels, testLabels, testNames, 3)
	count = count + 1
end
