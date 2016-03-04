local Threads = require 'threads'

local batchs = batchsz
local m = mean
local s = std
local configNames_tr, configNames_te  = configNames_train, configNames_test 
local configPathR = configPathRoot
local kk = kkk
local modelSaveR = modelSaveRoot
local mean_std_r = mean_std_root
local imgP = imgPath

donkeys = Threads(
         4,
         function()
            require 'torch'
         end,
         function(idx)
            batchSize = batchs -- pass to all donkeys via upvalue
            mean = m
            std = s
            tid = idx
			configNames_train, configNames_test = configNames_tr, configNames_te
			configPathRoot = configPathR 
			kkk = kk
			modelSaveRoot = modelSaveR
			mean_std_root = mean_std_r
			imgPath = imgP
            print(string.format('Starting donkey with id: %d', tid))
--            dofile('MyProgram/preprocess.lua')
            dofile('My/balance_pro.lua')
            dofile('My/util.lua')
 --           dofile('MyProgram/MySGD.lua')
         end
      );
