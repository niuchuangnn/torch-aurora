require 'nn'
require 'torch'
require 'cutorch'
dofile('My/path.lua')
dofile('My/balance_mean_std.lua')
dofile('My/util.lua')
dofile('My/getNL.lua')
dofile('My/MySGD.lua')
dofile('My/balance_pro.lua')
dofile('My/AlexNet_Model_c.lua')
dofile('My/train.lua')
dofile('My/test.lua')


for i = 1,55 do
   train()
   if i == 55 then
      test()
   end
   epoch = epoch + 1
end
