require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
dofile('My/util.lua')
dofile('My/CNN_features/parameters.lua')
dofile('My/CNN_features/load_mean_std.lua')
dofile('My/CNN_features/pathes.lua')
dofile('My/CNN_features/model_CNN.lua')
dofile('My/balance_pro.lua')
dofile('My/CNN_features/get_config_names.lua')
dofile('My/get_train_test_config.lua')
dofile('My/CNN_features/threads.lua')
dofile('My/CNN_features/extract_features.lua')

config_names_train, config_names_test = getTTConfigNames(configRoot)
model_names = getModelNames(modelRoot)

for i = 1, #model_names do
	print('start')
	cp_train = configRoot .. config_names_train[i]
	cp_test = configRoot .. config_names_test[i]
	mp = modelRoot .. model_names[i]
	mean, std = loadMS(config_names_train[i])
	print('start')
	--dofile('My/CNN_features/threads.lua')
	print('end')
	for n,k,v in string.gmatch(cp_train,"(%a+)_(%d+)_(%d+)") do
		sp_train = savePathRoot .. n .. '_' .. k .. '_' .. v
	end
	for n,k,v in string.gmatch(cp_test,"(%a+)_(%d+)_(%d+)") do
		sp_test = savePathRoot .. n .. '_' .. k .. '_' .. v
	end
	print('model:' .. mp)
	print('extract CNN features in ' .. config_names_train[i])
	extract_CNN(mp, cp_train, sp_train, mean, std)
	print('extract CNN features in ' .. config_names_test[i])
	extract_CNN(mp, cp_test, sp_test, mean, std)
--	donkeys:synchronize()
--	donkeys:terminate()
--	donkeys = nil
--	collectgarbage("collect")
end
