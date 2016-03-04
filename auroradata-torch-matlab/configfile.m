clear all

path_root = '/media/ljm/SSD2/Aurora201510/train_test_split_20150913';
N = dir(fullfile(path_root,'*.txt'));
N = struct2cell(N);
num = size(N);

if ~exist('/media/ljm/Data/config','dir')
    mkdir('/media/ljm/Data/config');
end
savePath = '/media/ljm/Data/config';

for i = 1:num(2)
    path = strcat(path_root,'/',N{1,i});
    fopen([savePath '/' N{1,i}],'a');
    fid = fopen([savePath '/' N{1,i}],'wt');
    [r,names,labels] = textread(path,'%d%s%d');
    for j = 1:length(names)
        filename = [names{j,1} '.jpg'];
        type = labels(j);
        fprintf(fid,'%s',filename);
        fprintf(fid,'%s',' ');
        fprintf(fid,'%d',type);
        fprintf(fid,'%s\n','');
    end
    fclose(fid);
end
    