-- Calculating mean and std of trianing set.
dofile('My/getNL.lua')
require 'image'
require 'torch'
--require 'cutorch'
--local imgPath = '/media/ljm/Data/Aurora_all/'
--local config = '/media/ljm/Data/config/train_15_4_1_2487.txt'
local config = trainConfig
local NL = getNL(config)
local totalsize = #NL[1] + #NL[2] + #NL[3] + #NL[4]
mean = 0
std = 0
print('Estimating mean and std ...')
for i = 1, 4 do
    for j = 1, #NL[i] do
        path = imgPath .. i .. '/' .. NL[i][j]
        im = image.load(path)
        im = image.crop(im,65,65,65+310,65+310)
        im = image.scale(im,256,256)
        mean = mean + im:mean()
        std = std + im:std()
    end
end

mean = mean/totalsize
std = std/totalsize

-- save mean and std
torch.save(mean_std_path,{mean = mean, std = std})

print('mean: ' .. mean)
print('std: ' .. std)
