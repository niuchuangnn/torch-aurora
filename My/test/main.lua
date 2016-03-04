require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
dofile('/home/ljm/torch/My/util.lua')
dofile('/home/ljm/torch/My/test/parameters.lua')
dofile('/home/ljm/torch/My/test/load_mean_std.lua')
dofile('/home/ljm/torch/My/test/pathes.lua')
dofile('/home/ljm/torch/My/test/model.lua')
dofile('/home/ljm/torch/My/balance_pro.lua')
dofile('/home/ljm/torch/My/test/get_config_names.lua')
dofile('/home/ljm/torch/My/get_train_test_config.lua')
dofile('/home/ljm/torch/My/test/threads.lua')
dofile('/home/ljm/torch/My/test/extract_features.lua')

config_names_train, config_names_test = getTTConfigNames(configRoot)
model_names = getModelNames(modelRoot)
confusionM = torch.Tensor(3,3):zero()
accuracy = torch.Tensor(1,3):zero()

for i = 1, #model_names do
	--print('start')
	--cp_train = configRoot .. [i]
	cp_test = configRoot .. config_names_test[i]
	mp = modelRoot .. model_names[i]
	mean, std = loadMS(config_names_train[i])
	--print('start')
	--dofile('My/CNN_features/threads.lua')
	--print('end')
	--for n,k,v in string.gmatch(cp_train,"(%a+)_(%d+)_(%d+)") do
	--	sp_train = savePathRoot .. n .. '_' .. k .. '_' .. v
	--end
	for n,k,v in string.gmatch(cp_test,"(%a+)_(%d+)_(%d+)") do
		sp_test = savePathRoot .. n .. '_' .. k .. '_' .. v
	end
	print('model:' .. mp)
	--print('extract CNN features in ' .. config_names_train[i])
	--extract_CNN(mp, cp_train, sp_train, mean, std)
	print('calculating accuracy of ' .. config_names_test[i])
	test(mp, cp_test, sp_test, mean, std)

      -- calculate accuracy
	  NUM = confusionM:sum(2)
	  for mm = 1,3 do
		accuracy[1][mm] = confusionM[mm][mm] / NUM[mm][1]
	  end

	  if not io.open(sp_test) then
	     local cmd = 'mkdir ' .. sp_test
		 sys.execute(cmd)
	  end
	  result = {confusionM = confusionM, accuracy = accuracy}
      mattorch.save(sp_test .. '/accuracy.mat',result)
	  print('confusionM: ')
	  print(confusionM)
	  print('accuracy: ')
	  print(accuracy)
	  confusionM:zero()
	  accuracy:zero()
--	donkeys:synchronize()
--	donkeys:terminate()
--	donkeys = nil
--	collectgarbage("collect")
end
