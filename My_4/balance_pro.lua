-- 1.Load images accroding to config files.
-- 2.Two modes 'train' and 'test' are provided, for 'train' mode, a 'balance' mode is alternative.
-- 3.The loadData function returns a batch of trianing or testing data which are preprocessed.
require 'image'
require 'torch'
--dofile('My/setPathes.lua')
dofile('/home/ljm/torch/My_4/getNL.lua')
--dofile('/home/ljm/torch/MyProgram/MyProgram/balance_mean_std.lua')
--mean = 0.157
--std = 0.0929
	
--local trainConfig = '/media/ljm/Data/config/train_15_4_1_2487.txt'
--local testConfig = '/media/ljm/Data/config/test_15_4.txt'
--local imgPath = '/home/ljm/AuroraImgData_gray/Aurora_img_4/'
if not CNN then
	print('not CNN')
	dofile('My/setPathes.lua')
	setPathes(kkk)
	print(trainConfig)
	print(testConfig)
	NL_train = getNL(trainConfig)
	NL_test = getNL(testConfig)
	trainsz = #NL_train[1] + #NL_train[2] + #NL_train[3] + #NL_train[4]
	testsz = #NL_test[1] + #NL_test[2] + #NL_test[3] + #NL_test[4]
	print('The number of train data is: ' .. trainsz)
	print('The number of test data is: ' .. testsz)
	batchsz = 256
	epochSize = math.ceil(trainsz/batchsz)
	testBatchsz = 20
	testBatch_n = math.ceil(testsz/testBatchsz)
	testList = {}
	testLabel = {}
	for i = 1,4 do
	    for j = 1,#NL_test[i] do
	        table.insert(testList,NL_test[i][j])
	        table.insert(testLabel,i)
	    end
	end
else
	local path_CNN = 'ini'
	local CNNList = {}
	local CNNLabel = {}
end

loadData = function(batchsz, mode, trainmode_or_st, en, path, mean, std)
assert(mode == 'train' or mode == 'test' or mode == 'CNN','train for loading train data, test for loading test data')
  if mode == 'train' then
    if trainmode_or_st == 'balance' then
      eachsize = batchsz/4
      local r = torch.Tensor(4,eachsize)
      r[1] = torch.randperm(#NL_train[1])[{ {1,eachsize} }]
      r[2] = torch.randperm(#NL_train[2])[{ {1,eachsize} }]
      r[3] = torch.randperm(#NL_train[3])[{ {1,eachsize} }]
      r[4] = torch.randperm(#NL_train[4])[{ {1,eachsize} }]
      local iw = 256
      local ih = 256
      local ow = 224
      local oh = 224
      local data = {}
      local outdata = {}
      local label = {}
      local outlabel = {}
      for i = 1, 4 do
          for j = 1, eachsize do
              path = imgPath .. i .. '/' .. NL_train[i][r[i][j]]
              im = image.load(path)
--              im = image.crop(im,65,65,65+310,65+310)
              im = image.scale(im,256,256)
              im:add(-mean)
              im:div(std)
              -- do hflip with probability 0.5
              if torch.uniform() > 0.5 then im = image.hflip(im) end
              
              -- do random crop
              local h1 = math.ceil(torch.uniform(1e-2, ih-oh))
              local w1 = math.ceil(torch.uniform(1e-2, iw-ow))
              table.insert(data,image.crop(im,w1,h1,w1+ow,h1+ow))
              table.insert(label, i)
          end
       end
       rr = torch.randperm(batchsz)
       for i = 1, batchsz do
           table.insert(outdata,data[rr[i]])
           table.insert(outlabel,label[rr[i]])
       end
       outputs = torch.Tensor(#outdata,1,ow,oh):float()
       labels = torch.Tensor(#outlabel)
       for j = 1,#outdata do
           outputs[{  {j},{},{},{} }] = outdata[j]:float()
           labels[j] = outlabel[j]
       end
       data = nil
       outdata = nil
       label = nil
       outlabel = nil
    end
  end

  if mode == 'test' then
    local b1 = trainmode_or_st
    local b2 = en
    
    local iw = 256
    local ih = 256
    local ow = 224
    local oh = 224
    local data = {}
    local outdata = {}
    local label = {}

    for i = b1,b2 do
        table.insert(label,testLabel[i])
        path = imgPath .. testLabel[i] .. '/' .. testList[i]
        im = image.load(path)
--        im = image.crop(im,65,65,65+310,65+310)
        im = image.scale(im,256,256)
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
    labels = torch.Tensor(b2 - b1 + 1)
    for i = 1,b2-b1+1 do
        labels[i] = label[i]
    end
    data = nil
    outdata = nil
    label = nil
  end

  if mode == 'CNN' then
    local b1 = trainmode_or_st
    local b2 = en
	if path_CNN ~= path then
		print('true')
		path_CNN = path
    	local NL_CNN = getNL(path_CNN)
		CNNList = {}
		CNNLabel = {}
		for i = 1,4 do
    		for j = 1,#NL_CNN[i] do
        		table.insert(CNNList,NL_CNN[i][j])
				table.insert(CNNLabel,i)
   	 		end
		end
	end
    --local iw = 256
    --local ih = 256
    local ow = 224
    local oh = 224
    local data = {}
    local outdata = {}
    local label = {}
	names = {}
    for i = b1,b2 do
		table.insert(label,CNNLabel[i])
        local path_abs = imgPath .. CNNLabel[i] .. '/' .. CNNList[i]
        im = image.load(path_abs)
--        im = image.crop(im,65,65,65+310,65+310)
        im = image.scale(im,ow,oh)
        -- mean/std
        im:add(-mean)
        im:div(std)
        table.insert(outdata,im)   
		table.insert(names,CNNList[i])   
    end
    outputs = torch.Tensor(#outdata,1,ow,oh):float()

    for j = 1,#outdata do
       outputs[{  {j},{},{},{} }] = outdata[j]:float()
    end
    labels = torch.Tensor(b2 - b1 + 1)
    for i = 1,b2-b1+1 do
        labels[i] = label[i]
    end
    data = nil
    outdata = nil
    label = nil
	collectgarbage("collect")
	return	outputs, labels:float(), names
  end

    collectgarbage("collect")
    return outputs, labels:float()
end


		


