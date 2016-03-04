require 'nn'
require 'torch'
require 'cutorch'
require 'cunn'
--dofile('My/path.lua')
--dofile('My/balance_mean_std.lua')
dofile('My/util.lua')
dofile('My/getNL.lua')
dofile('My/MySGD.lua')
--dofile('My/balance_pro.lua')
dofile('My/AlexNet_Model_c.lua')

dofile('My/get_train_test_config.lua')
dofile('My/setPathes.lua')

--CNN = false
-- create model
model = createModel()

-- create criterion
criterion = nn.ClassNLLCriterion()

print('==> Model')
print(model)

print('==> criterion')
print(criterion)

-- convert model to CDUA
print('==> converting model to CUDA')
model:cuda()
criterion:cuda()

modelSaveRoot = '/media/ljm/Data/Experiments_3_20160229/models/'
configPathRoot = '/media/ljm/Data/Experiments_3_20160229/config1/'
mean_std_root = '/media/ljm/Data/Experiments_3_20160229/mean_std/'
imgPath = '/home/ljm/NiuChuang/AuroraData_gray/Aurora_img/'
configNames_train, configNames_test = getTTConfigNames(configPathRoot)
--	kkk = 1
--	setPathes(kkk)
--	print(trainConfig)
--	dofile('My/balance_mean_std.lua')
--	dofile('My/balance_pro.lua')
--	dofile('My/threads.lua')
dofile('My/train.lua')
dofile('My/test.lua')


for iii = 1, #configNames_train do
--	iii = #configNames_train	
--	trainConfig = configPathRoot .. configNames_train[kkk]
--	testConfig = configPathRoot .. configNames_test[kkk]
--    modelpath = modelSaveRoot .. 'model' .. string.sub(configNames_train[kkk], 6, 10) .. '.bin'
--	mean_std_path = mean_std_root .. 'mean_std' .. string.sub(configNames_train[kkk], 6, 10) .. '.bin'
	kkk = iii
	setPathes(kkk)
	dofile('My/balance_mean_std.lua')
	dofile('My/balance_pro.lua')
	dofile('My/threads.lua')
	for i = 1,55 do
		train()
		if i == 55 then
			test()
   		end
   		epoch = epoch + 1
	end
	epoch = nil
	donkeys:terminate()
	donkeys = nil
	collectgarbage("collect")
	model:reset()
end
