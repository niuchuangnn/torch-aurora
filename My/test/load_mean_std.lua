-- This function return mean and std according to the given Config path

function loadMS(configPath)
	local msp = ''
	for m,k,v in string.gmatch(configPath,"(%a+)_(%d+)_(%d+)") do
		msp = mean_std_root .. '/mean_std_' .. k .. '_' .. v .. '.bin'
	end
	ms = torch.load(msp)
	return ms.mean, ms.std
end
