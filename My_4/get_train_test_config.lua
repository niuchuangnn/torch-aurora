-- This function return train and test config file names under the given directory.

function getTTConfigNames(path)
	local configNames_str = sys.ls(path)
	local config_names = {}
	local configNames_train = {}
	local configNames_test = {}
	local min = 0
	local vi = 0
	local vj = 0
	local tmp = 'tmp'
	local min_v = 0

	for k, v in string.gmatch(configNames_str, "([%w_]+)%.(%w+)") do
		table.insert(config_names, k .. '.' .. v)
	end
	
	for i = 1, #config_names do
		for k, v in string.gmatch(config_names[i], "(%a+)_(%d+)") do
			vi = v + 0
		end
		min = i
		min_v = vi
		for j = i + 1, #config_names do
			for k, v in string.gmatch(config_names[j], "(%a+)_(%d+)") do
				vj = v + 0
			end
			if vj+0 < min_v then
				min = j
				min_v = vj 
			end
		end
		tmp = config_names[min]
		config_names[min] = config_names[i]
		config_names[i] = tmp
	end

	for i = 1, #config_names do
		if string.sub(config_names[i],1,5) == 'train' then
			table.insert(configNames_train,config_names[i])
		elseif string.sub(config_names[i],1,4) == 'test' then
			table.insert(configNames_test,config_names[i])
		else
			print('Error config name:' .. config_names[i])
		end
	end

	return configNames_train, configNames_test
end
