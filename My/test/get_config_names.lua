-- This function return config file names under the given directory.

function getConfigNames(path)
	local configNames_str = sys.ls(path)
	local configNames = {}
	for k, v in string.gmatch(configNames_str, "([%w_]+)%.(%w+)") do
		table.insert(configNames, k .. '.' .. v)
	end
	return configNames
end

