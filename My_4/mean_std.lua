require 'torch'
require 'hdf5'

local f = hdf5.open('/home/ljm/torch/auroraData/traindata.h5','r')

-- Estimate the mean and std
-- shuffle = torch.randperm(trainsz)
mean = 0
std = 0
print('Estimating mean and std ...') 
for i = 1,10000,200 do
    local samples = f:read('/data'):partial({1,256}, {1,256}, {1,1}, {i,i+200-1})
    samples = samples:transpose(1,4)
    samples = samples:transpose(2,3)
    for j = 1,200 do
        mean = mean + samples[j][1]:mean()
        std = std + samples[j][1]:std()
    end
end
mean = mean/10000
std = std/10000
print(mean)
print(std)
