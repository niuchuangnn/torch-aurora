require 'threads'

local Threads = require 'threads'
--local m = mean
--local s = std
local batchs = Batchsz
local c = CNN
local imgP = imgPath
donkeys = Threads(
         4,
         function()
            require 'torch'
         end,
         function(idx)
            batchSize = batchs -- pass to all donkeys via upvalue
			imgPath = imgP
            mean = m
            std = s
			CNN = c
            tid = idx
            print(string.format('Starting donkey with id: %d', tid))
--            dofile('MyProgram/preprocess.lua')
            dofile('My/balance_pro.lua')
            dofile('My/util.lua')
--           dofile('MyProgram/MySGD.lua')
         end
      );
