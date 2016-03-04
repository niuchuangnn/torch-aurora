dofile('My/CNN_features/model_CNN.lua')
local savePath = '/media/ljm/Data/Experiments/models_cpu/'
local modelRoot = '/media/ljm/Data/Experiments/models/'

local modelNames = getModelNames(modelRoot)
for i = 1, #modelNames do
	m_g = loadCNNModel(modelRoot .. modelNames[i])
	m_c = m_g:float()
--	p,g = m_c:getParameters()
	torch.save(savePath .. modelNames[i], m_c)
	print(modelNames[i] .. ' saved')
end


