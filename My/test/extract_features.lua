require 'torch'
require 'xlua'
require 'optim'
require 'cutorch'
require 'mattorch'
require 'threads'
print '==> defining extrcting CNN features procedure'
--local Batchsz = 256
--local Batch_n = math.ceil(Allsz/Batchsz)

-- This function extracts CNN features of a given config file(absolute config file path) with a given trained CNN model(absolute model path), which are saved into a given directory.



function test(modelPath, config, savePath, mean, std)

   local time = sys.clock()
   cutorch.synchronize()
   
   -- load CNN model
   --local model = loadCNNModel(modelPath)
   --model:evaluate()

   -- get model parameters
   --parameters_cpu, gradParameters_cpu = model:getParameters()

   --print('parameters size on cpu: ')
   --print(parameters_cpu:size())
   --print('gradParameters size on gpcpu: ')
   --print(gradParameters_cpu:size())
   -- copy to gpu
   --parameters:copy(parameters_cpu)
   --gradParameters:copy(gradParameters_cpu)
--print(cutorch.getMemoryUsage())
--model_CNN = loadCNNModel('/media/ljm/Data/Experiments/models/model_7_12.bin')
model = loadModel(modelPath)
-- convert model to cuda
model = model:cuda()
-- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
model:evaluate()
-- cuda tensor
parameters, gradParameters = model:getParameters()
--print('parameters size on gpu: ')
--print(parameters:size())
--print('gradParameters size on gpu: ')
--print(gradParameters:size())
--print(cutorch.getMemoryUsage())
   -- extract CNN features from a config file
   print('==> testing accuracy on ' .. config)
   local _, configsz = getNL(config)
   local Batch_n = math.ceil(configsz/Batchsz)
   for t = 1, Batch_n do
      local indexStart = (t-1) * Batchsz + 1
      local indexEnd = math.min(t*Batchsz,configsz)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels, names = loadData(Batchsz,'CNN',indexStart, indexEnd, config, mean, std)
            return sendTensor(inputs), sendTensor(labels), names, savePath
         end,
         -- callback that is run in the main thread once the work is done
         extractBatch
      )
      if t % Batch_n == 0 then
         donkeys:synchronize()
         collectgarbage()
      end

      -- disp progress
      xlua.progress(t, Batch_n) 
   end

   donkeys:synchronize()
   cutorch.synchronize()

   -- timing
   time = sys.clock() - time
   print("\n==> time to extract " .. config .. ': ' .. configsz .. ' : ' .. time .. 's')
end
-------------------------------------------------------------------------------------
-- extract batch

local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.FloatTensor()
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function extractBatch(inputsThread, labelsThread, names,savePath)


   receiveTensor(inputsThread, inputsCPU)
   receiveTensor(labelsThread, labelsCPU)
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local cont = labels:size()[1]
   for i = 1,cont do

      -- evaluate confusion matrix
      local oo =  model:forward(inputs[i])
	  _,index = oo:sort()
	  confusionM[labels[i]][index[3]] = confusionM[labels[i]][index[3]] + 1

      cutorch.synchronize()
   end
--print(cutorch.getMemoryUsage())
end







