-- This function returns Names Table and Labels Table.
dofile('/home/ljm/torch/My/getNL.lua')
function getNLTable(config)
	local NL = getNL(config)
	local namesTable = {}
	local labelsTable = {}
	for i = 1,3 do
	    for j = 1,#NL[i] do
	        table.insert(namesTable,string.sub(NL[i][j], 1, -5))
	        table.insert(labelsTable,i)
	    end
	end
	return namesTable, labelsTable
end
