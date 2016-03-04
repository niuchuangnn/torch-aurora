-- This function calculate the accuracy by kNN algorithm according to the given distance matrix, train Lables and test Labels.
require 'mattorch'

function kNN(distanceM, trainLabels, testLabels, testNames, k)
	local trainSize = #trainLabels
	local testSize = #testLabels
	local accuracy_num = torch.Tensor(3):zero()
	local accuracy = torch.Tensor(3):zero()
	local type_num = torch.Tensor(3):zero()
	local confusionM = torch.Tensor(3,3):zero()

	for i = 1, 3 do
		type_num[i] = #testNames[i]
		print('type ' .. i .. ': ' .. #testNames[i])
	end

	print('calculating accuracy ...')
	for i = 1, testSize do
		sorted, index = distanceM[{ {}, {i} }]:sort(1)
		-- 1NN
		if k == 1 then
			confusionM[{ {testLabels[i]}, {trainLabels[index[1][1]]} }] = confusionM[{ {testLabels[i]}, {trainLabels[index[1][1]]} }] + 1

			if trainLabels[index[1][1]] == testLabels[i] then
				accuracy_num[testLabels[i]] = accuracy_num[testLabels[i]] + 1
			end
		end

		if k == 3 then
			local knn_3_type_num = torch.Tensor(3):zero()
			local knn_3 = torch.Tensor(3):zero()

			for j = 1,3 do
				knn_3[j] = trainLabels[index[j][1]]
				knn_3_type_num[knn_3[j]] = knn_3_type_num[knn_3[j]] + 1
			end
			-- 3NN
			local t, ni = knn_3_type_num:sort()

			confusionM[{ {testLabels[i]}, ni[3] }] = confusionM[{ {testLabels[i]}, ni[3] }] + 1
			if ni[3] == testLabels[i] then

				accuracy_num[testLabels[i]] =accuracy_num[testLabels[i]] + 1
			end
		end
	end
	
	for i = 1, 3 do
		assert(#testNames[i] == confusionM[{ {i}, {} }]:sum(), 'confusion matrix fault')
--		print(#testNames[i])
--		print(confusionM[{ {i}, {} }]:sum())
	end

	print(confusionM)
	print('accuracy based on confusion matrix:')
	for i = 1, 3 do
		nn = (confusionM[{ {i}, {i} }] / (confusionM[{ {i}, {} }]:sum()))
		accuracy[i] = confusionM[{ {i}, {i} }] / confusionM[{ {i}, {} }]:sum()
	end
	print('average accuracy: ' .. accuracy:mean())

	print('accuracy based on right predicted counts: ')
	for i = 1, 3 do
		iii = accuracy_num[i] / type_num[i]
		accuracy[i] = accuracy_num[i] / type_num[i]
	end
	print('average accuracy: ' .. accuracy:mean())
	count =  count or 7
	local AC = {}
	local an_a = 'train_' .. count .. '.mat'
	AC.confusion = confusionM
	AC.accuracy = accuracy
	mattorch.save(an_a,AC)
	
end
