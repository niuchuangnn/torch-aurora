require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'cutorch'
require 'cunn'
require 'threads'


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

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

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

batchSize = 128

print '==> defining training procedure'

function train()

   local time = sys.clock()

   -- epoch tracker
   epoch = epoch or 1

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   local params, new = paramsForEpoch(epoch)
   optimState.learningRate = params.learningRate
   optimState.weightDecay = params.weightDecay
   optimState.new = new

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,trainsz,batchSize do
      -- disp progress
      xlua.progress(t, trainsz)
      
      batch_num = (t-1)/batchSize + 1

      -- create mini batch
      -- print('==>loading mini batch train data')
      local inputs,targets = loadData(batchSize,batch_num,'train')
      targets = targets + 1
      for j = 1,#inputs do
         inputs[j] = inputs[j]:float()
         inputs[j] = inputs[j]:cuda()
      end
      targets = targets:cuda()

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
         optimMethod(feval, parameters, optimState)
         if optimState.new then
            optimState.new = false
         end
   end

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




























