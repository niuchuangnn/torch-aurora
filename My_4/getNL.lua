-- read config file and save filenames and corrresponding labels into a tabel
function getNL(path)
    --path = '/home/ljm/NiuChuang/AuroraData_gray/Experiment2/Exp1_aurora_val.text'
    local f1 = io.input(path)
    local str = io.read('*a')
    local NL = {}
    local type1 = {}
    local type2 = {}
    local type3 = {}
    local type4 = {}
    local s = torch.Tensor(1,4):zero()

    for i = 1,string.len(str),23 do
        local name = string.sub(str,i,i+19)
        local label = string.sub(str,i+20,i+21)
        if label+0 == 1 then
           table.insert(type1,name)
           s[1][1] = s[1][1] + 1
        end
        if label+0 == 2 then
           table.insert(type2,name)
           s[1][2] = s[1][2] + 1
        end
        if label+0 == 3 then
           table.insert(type3,name)
           s[1][3] = s[1][3] + 1
        end
        if label+0 == 4 then
           table.insert(type4,name)
           s[1][4] = s[1][4] + 1
        end
    end
    table.insert(NL,type1)
    table.insert(NL,type2)
    table.insert(NL,type3)
    table.insert(NL,type4)
    for i = 1,4 do
        print("type" .. i .. ": " .. s[1][i])
    end
    print("total: " .. s:sum())
    return NL, s:sum()
end
