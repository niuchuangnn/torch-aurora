--
-- Load data and do some preprocess

-- batchsz:  the number of training/testing examples to be loaded
-- count:  the count' batch
-- mode:  train 0r test

require 'torch'
require 'image'
require 'hdf5'

--local count = 0
local f = hdf5.open('/home/ljm/torch/auroraData/traindata.h5','r')
local f1 = hdf5.open('/home/ljm/torch/auroraData/testdata.h5','r')
local im = torch.Tensor(1,256,256)
trainsz = f:read('/data'):dataspaceSize()[4]
testsz = f1:read('/data'):dataspaceSize()[4]
print('The number of train data is: ' .. trainsz)
print('The number of test data is: ' .. testsz)
batchsz = 256
epochSize = math.ceil(trainsz/batchsz)

loadData = function(batchsz, mode, st, en)
assert(mode == 'train' or mode == 'test' or mode == 'AE','train for loading train data, test for loading test data')


-- load train data
  if mode == 'train' then

--    if count == 0 then
--       count = count + 1
--    end

--   if count > epochSize then
--       count = 1
--   end
    count = torch.ceil(torch.uniform(1e-12,epochSize))
    print('loading ' .. count .. ' batch data')
    b1 = (count-1)*batchsz+1
    b2 = math.min(count*batchsz,trainsz)
    data = f:read('/data'):partial({1,256}, {1,256}, {1,1}, {b1,b2})
    data = data:transpose(1,4)
    data = data:transpose(2,3)
    label = f:read('/label'):partial({1,1}, {b1,b2})
    label = label[1] + 1
    outdata = {}
    
    local iw = 256
    local ih = 256
    local ow = 224
    local oh = 224
    for i = 1,b2 - b1 + 1 do
        im[{  {},{},{} }] = data[{ {i}, {1}, {1,256}, {1,256} }]
        -- mean/std
        im:add(-mean)
        im:div(std)
        -- do hflip with probability 0.5
        if torch.uniform() > 0.5 then im = image.hflip(im) end
        -- do random crop
        local h1 = math.ceil(torch.uniform(1e-2, ih-oh))
        local w1 = math.ceil(torch.uniform(1e-2, iw-ow))
        table.insert(outdata,image.crop(im,w1,h1,w1+ow,h1+ow))
    end
    outputs = torch.Tensor(#outdata,1,ow,oh):float()
    for j = 1,#outdata do
       outputs[{  {j},{},{},{} }] = outdata[j]:float()
    end
--    count = count + 1
  end
-- load test data
  if mode == 'test' then
    b1 = st
    b2 = en
    data = f1:read('/data'):partial({1,256}, {1,256}, {1,1}, {b1,b2})
    data = data:transpose(1,4)
    data = data:transpose(2,3)
    label = f1:read('/label'):partial({1,1}, {b1,b2})
    label = label[1] + 1
    outdata = {}
    
    local iw = 256
    local ih = 256
    local ow = 224
    local oh = 224
    for i = 1,b2 - b1 + 1 do
        im[{  {},{},{} }] = data[{ {i}, {1}, {1,256}, {1,256} }]
        -- mean/std
        im:add(-mean)
        im:div(std)
        local w1 = math.ceil((iw-ow)/2)
        local h1 = math.ceil((ih-oh)/2)
        table.insert(outdata,image.crop(im,w1,h1,w1+ow,h1+ow)) -- center patch
        table.insert(outdata,image.hflip(image.crop(im,w1,h1,w1+ow,h1+ow)))
        h1 = 1; w1 = 1;
        table.insert(outdata,image.crop(im,w1,h1,w1+ow,h1+ow)) -- top-left patch 
        table.insert(outdata,image.hflip(image.crop(im,w1,h1,w1+ow,h1+ow)))
        h1 = 1; w1 = iw - ow;
        table.insert(outdata,image.crop(im,w1,h1,w1+ow,h1+ow)) -- top-right patch 
        table.insert(outdata,image.hflip(image.crop(im,w1,h1,w1+ow,h1+ow)))      
        h1 = ih - oh; w1 = 1;
        table.insert(outdata,image.crop(im,w1,h1,w1+ow,h1+ow)) -- bottom-left patch 
        table.insert(outdata,image.hflip(image.crop(im,w1,h1,w1+ow,h1+ow)))      
        h1 = ih - oh; w1 = iw - ow;
        table.insert(outdata,image.crop(im,w1,h1,w1+ow,h1+ow)) -- bottom-right patch 
        table.insert(outdata,image.hflip(image.crop(im,w1,h1,w1+ow,h1+ow)))      
    end
    outputs = torch.Tensor(#outdata,1,ow,oh):float()
    for j = 1,#outdata do
       outputs[{  {j},{},{},{} }] = outdata[j]:float()
    end
  end
  -- AutoEncoder mode
  if mode == 'AE' then
    count = torch.ceil(torch.uniform(1e-12,epochSize))
    print('loading ' .. count .. ' batch data')
    b1 = (count-1)*batchsz+1
    b2 = math.min(count*batchsz,trainsz)
    data = f:read('/data'):partial({1,256}, {1,256}, {1,1}, {b1,b2})
    data = data:transpose(1,4)
    data = data:transpose(2,3)
    outdata = {}
    
    local iw = 256
    local ih = 256
    local ow = 224
    local oh = 224
    for i = 1,b2 - b1 + 1 do
        im[{  {},{},{} }] = data[{ {i}, {1}, {1,256}, {1,256} }]
        -- mean/std
        im:add(-mean)
        im:div(std)
        -- do hflip with probability 0.5
        if torch.uniform() > 0.5 then im = image.hflip(im) end
        -- do random crop
        local h1 = math.ceil(torch.uniform(1e-2, ih-oh))
        local w1 = math.ceil(torch.uniform(1e-2, iw-ow))
        table.insert(outdata,image.crop(im,w1,h1,w1+ow,h1+ow))
    end
    outputs = torch.Tensor(#outdata,1,ow,oh):float()
    for j = 1,#outdata do
       outputs[{ {j},{},{},{} }] = outdata[j]:float()
    end
    label = outputs:clone()
  end     
    return outputs, label
end












