require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'

local min = 0
local vi = 0
local vj = 0
local tmp = 'tmp'
local min_v = 0

function getModelNames(path)
	local names_str = sys.ls(path)
	names = {}
	for i = 1, names_str:len(), 15 do
		table.insert(names, string.sub(names_str, i, i + 13))
	end

	for i = 1, #names do
		for k, v in string.gmatch(names[i], "(%a+)_(%d+)") do
			vi = v + 0
		end
		min = i
		min_v = vi
		for j = i + 1, #names do
			for k, v in string.gmatch(names[j], "(%a+)_(%d+)") do
				vj = v + 0
			end
			if vj+0 < min_v then
				min = j
				min_v = vj 
			end
		end
		tmp = names[min]
		names[min] = names[i]
		names[i] = tmp
	end

	return names
end

function loadCNNModel(path)
	print('Loading model ...')
	local model = torch.load(path)
	model_CNN = model.modules[1]
	return model_CNN
end
