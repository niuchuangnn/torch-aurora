require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'cutorch'
require 'cunn'
require 'threads'


local Threads = require 'threads'
local withDevice = cutorch.withDevice
local batchs = batchsz
local m = mean
local s = std
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
            print(string.format('Starting donkey with id: %d', tid))
--            dofile('MyProgram/preprocess.lua')
            dofile('My/balance_pro.lua')
            dofile('My/util.lua')
 --           dofile('MyProgram/MySGD.lua')
         end
      );

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

-- GPU 2
withDevice(2,function() criterion1 = criterion:clone() end)

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

-- GPU 2
withDevice(2,function()
                 model1 = model:clone()
                 parameters1,gradParameters1 = model1:getParameters()
             end)

print '==> configuring optimizer'
   local optimState = {
    learningRate = 0.01,
    learningRateDecay = 0.0,
    momentum = 0.9,
    dampening = 0.0,
    weightDecay = 5e-4,
    new = false
}
optimMethod = MySGD

function paramsForEpoch(epoch)
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     18,   1e-2,   5e-4, },
        { 19,     29,   5e-3,   5e-4  },
        { 30,     43,   1e-3,   0 },
        { 44,     52,   5e-4,   0 },
        { 53,    1e8,   1e-4,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end
-----------------------------------------------------------------------
-- trian

-- classes
classes = {'1','2','3','4'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger('MyProgram/train.log')
testLogger = optim.Logger('MyProgram/test.log')


epochSize = math.ceil(trainsz/batchsz)

print '==> defining training procedure'

function train()

   local time = sys.clock()

   -- epoch tracker
   epoch = epoch or 1
--   local batchNumber = 0

   local params, new = paramsForEpoch(epoch)
   optimState.learningRate = params.learningRate
   optimState.weightDecay = params.weightDecay
   optimState.new = new

   cutorch.synchronize()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchsz .. ']')

   for i=1,epochSize do
--      local upvalue = upvalue + 1
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
--            queuevalue = upvalue
            local inputs, labels = loadData(batchSize,'train','balance')
 --           print(string.format('/n %d batch finished (ran on thread ID %x)',queuevalue, __threadid))
            return sendTensor(inputs), sendTensor(labels)
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
      if i % epochSize == 0 then
         donkeys:synchronize()
      end
   end

   donkeys:synchronize()
   cutorch.synchronize()


   -- time taken
   time = sys.clock() - time
   print("\n==> time to learn 1 epoch = " .. time .. 's')

   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   
   -- plot
   trainLogger:style{['% mean class accuracy (train set)'] = '-'}
   trainLogger:plot()


   confusion:zero()
end

-------------------------------------------------------------------------------------------
-- create tensor buffers in main thread and deallocate their storages.
-- the thread loaders will push their storages to these buffers when done loading
local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.FloatTensor()

-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

-- GPU 2 inputs
local inputs1 = torch.CudaTensor()
local labels1 = torch.CudaTensor()

-- Buffer serves to transfer parameters between two gpus
local buffer_parameters = torch.FloatTensor()
local buffer_gradParameters = torch.FloatTensor()

local t = 0

-- trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsThread, labelsThread)
   cutorch.synchronize()

   -- set the data and labels to the main thread tensor buffers (free any existing storage)
   receiveTensor(inputsThread, inputsCPU)
   receiveTensor(labelsThread, labelsCPU)

   -- transfer over to GPU 1(parent GPU)
   withDevice(1,function()
                   inputs:resize(inputsCPU:size()):copy(inputsCPU)
                   labels:resize(labelsCPU:size()):copy(labelsCPU)
                end)
   -- transfer over to GPU 2(parent GPU)
   withDevice(2,function()
                   inputs1:resize(inputsCPU:size()):copy(inputsCPU)
                   labels1:resize(labelsCPU:size()):copy(labelsCPU)
                end)


   if t == 0 then 
      t = t + 1
   end
   if t > epochSize then
      t = 1
   end
      -- disp progress
      xlua.progress(t, epochSize)
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- GPU 2
                       buffer_parameters:resize(parameters:size()):copy(parameters)
                       withDevice(2,function() parameters1:copy(buffer_parameters) end)

                       -- reset gradients
                       gradParameters:zero()
                       
                       -- reset gradients on GPU 2
                       withDevice(2,function() gradParameters1:zero() end)

                       -- f is the average of all criterions
                       local f = 0
                       local f1 = 0

                       -- evaluate function for complete mini batch
                       -- First half batch are trained on GPU 1, the second half on GPU 2
                       -- GPU 1(parent GPU)
                       withDevice(1,function()  
                                       for i = 1,inputs:size()[1]/2 do
                                           -- estimate f
                                           local output = model:forward(inputs[i])
                                           local err = criterion:forward(output, labels[i])
                                           f = f + err

                                           -- estimate df/dW
                                           local df_do = criterion:backward(output, labels[i])
                                           model:backward(inputs[i], df_do)

                                           -- update confusion
                                           confusion:add(output, labels[i])
                                       end
                                    end)

                       -- GPU 2
                       withDevice(2,function()  
                                       for j = inputs1:size()[1]/2 + 1,inputs1:size()[1] do
                                           -- estimate f1
                                           local output1 = model1:forward(inputs1[j])
                                           local err1 = criterion1:forward(output1, labels1[j])
                                           f1 = f1 + err1

                                           -- estimate df/dW
                                           local df_do1 = criterion1:backward(output1, labels1[j])
                                           model:backward(inputs1[j], df_do1)

                                           -- update confusion
                                           confusion:add(output1, labels1[j])
                                       end
                                    end)

                       cutorch.synchronize()
                       -- transfer gradParameters on GPU 2 to buffer
                       buffer_gradParameters = gradParameters:float()

                       -- merger gradParameters on two GPUs and normalize gradients and f(X)
                       gradParameters = (gradParameters + buffer_gradParameters:cuda()):div(inputs:size()[1])
                       f = (f + f1)/inputs:size()[1]

                       -- return f and df/dX
                       return f,gradParameters
                    end

         -- optimize on current mini-batch
         optimMethod(feval, parameters, optimState)

         -- transfer parameters on GPU 1 to GPU 2
         buffer_parameters:resize(parameters):copy(parameters)
         withDevice(2,function() parameters:copy(buffer_parameters()) end)
         if optimState.new then
            optimState.new = false
         end
   t = t + 1
   --print(cutorch.getDevice())
   --print(cutorch.getMemoryUsage())
end

























