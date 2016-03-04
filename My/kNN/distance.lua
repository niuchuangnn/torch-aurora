-- This function calculates the distance matrix between the given train and test features.
require 'mattorch'
require 'cutorch'
dofile('/home/ljm/torch/My/getNL.lua')
dofile('/home/ljm/torch/My/kNN/getNLTable.lua')
dofile('/home/ljm/torch/My/kNN/setPathes.lua')

function distance(trainFeatureConfig, testFeatureConfig, trainFM, testFM, distanceM)
	local trainNames, trainLabels = getNLTable(configRoot .. trainFeatureConfig)
	local testNames, testLabels = getNLTable(configRoot .. testFeatureConfig)
	local trainFeaturesFolder = featuresPathRoot .. getTTF(trainFeatureConfig) .. '/'
	local testFeaturesFolder = featuresPathRoot .. getTTF(testFeatureConfig) .. '/'
	local trainsz = #trainNames
	local testsz = #testNames
	local tN = getNL(configRoot .. testFeatureConfig)
--	local tp = trainFeaturesFolder .. trainNames[1] .. '.mat'
--	local t = mattorch.load(tp).cnn
--	local dimensionsz = t:size()[1] * t:size()[2] * t:size()[3]

	-- The features are saved into corresponding rows of train/testFeatureMatrix
--	local trainFeatureMatrix = torch.Tensor(trainsz, dimensionsz):float()
--	local testFeatureMatrix = torch.Tensor(testsz, dimensionsz):float()
	-- Dimension of diatance matrix is #train-set * #test-set
--	local distanceMatrix = torch.Tensor(trainsz, testsz):float()
	-- Load train and test features into feature matrix respectively
	print('Loading features ...')
	loadFeatures(trainNames, trainFeaturesFolder, trainFM)
	loadFeatures(testNames, testFeaturesFolder, testFM)
	-- convert matrixes to cuda
--	trainFeatureMatrix = trainFeatureMatrix:cuda()
--	testFeatureMatrix = testFeatureMatrix:cuda()
--	distanceMatirx = distanceMatrix:cuda()
	-- calculate distance
	calDistance(trainFM, testFM, distanceM, trainsz, testsz)
	return trainLabels, testLabels, tN
end

function getTTF(configName)
	local TF
	for n,k,v in string.gmatch(configName,"(%a+)_(%d+)_(%d+)") do
		TF = n .. '_' .. k .. '_' .. v
	end
	return TF
end

function loadFeatures(names, folder, matrix)
	for i = 1, #names do
		print('Loading ' .. folder .. names[i] .. '.mat')
		local f = mattorch.load(folder .. names[i] .. '.mat').cnn:reshape(128, 6, 6):reshape(128*6*6)
		matrix[{ {i}, {}}] = f
	end
--	return matrix
end

function calDistance(trainM,testM,distanceM,trainSz,testSz)
	local epsilon = 1e-12
	for i = 1, trainSz do
		print(trainSz .. ': ' .. i)
		for j = 1, testSz do
			distanceM[{ {i}, {j} }] = ((trainM[{ {i}, {} }] - testM[{ {j}, {} }]):cmul(trainM[{ {i}, {} }] - testM[{ {j}, {} }]):cdiv(trainM[{ {i}, {} }] + testM[{ {j}, {} }] + epsilon)):sum()
		end
	end
end
