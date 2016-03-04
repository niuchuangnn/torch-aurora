require 'torch'
require 'xlua'
require 'optim'
require 'cutorch'

print '==> defining test procedure'

--local testBatchsz = 20
--local testBatch_n = math.ceil(testsz/testBatchsz)

-- test function
function test()

   local time = sys.clock()
   
   cutorch.synchronize()
   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')



   for t = 1,testBatch_n do
      local indexStart = (t-1) * testBatchsz + 1
      local indexEnd = math.min(t*testBatchsz,testsz)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = loadData(testBatchsz,'test',indexStart, indexEnd,0,mean,std)
            return sendTensor(inputs), sendTensor(labels)
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
      if t % testBatch_n == 0 then
         donkeys:synchronize()
         collectgarbage()
      end

      -- disp progress
      xlua.progress(t, testBatch_n) 
   end

   donkeys:synchronize()
   cutorch.synchronize()

   -- timing
   time = sys.clock() - time
   print("\n==> time to test all samples " .. testsz .. ' : ' .. time .. 's')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   -- plot
   testLogger:style{['% mean class accuracy (test set)'] = '-'}
   testLogger:plot()

   -- next iteration:
   confusion:zero()
end
-------------------------------------------------------------------------------------
-- test batch

local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.FloatTensor()
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()


function testBatch(inputsThread, labelsThread)

   receiveTensor(inputsThread, inputsCPU)
   receiveTensor(labelsThread, labelsCPU)
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)
--   print(inputs:size())
   local cont = labels:size()[1]
   for i = 1,cont do

      -- test sample
      local pred =  model:forward(inputs[(i-1)*10 + 1])
      for j = 2,10 do
         cc = (i-1)*10 + j
         pred = pred + model:forward(inputs[cc])
      end
      cutorch.synchronize()  
      confusion:add(pred, labels[i])
   end

end







