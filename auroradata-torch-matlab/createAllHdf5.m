% transform all image data and its corresponding labels into a hdf5 file.

clear all

%% get all images' names and their corresponding types
allLPath = '/media/ljm/SSD2/Aurora201510/Alllabel2003_38044.txt';
imgPath = '/media/ljm/SSD2/Aurora201510/labeled2003_38044_201510/';
hdf5SavePath = '/media/ljm/Data/allData.h5';

[filenames types] = textread(allLPath,'%s%d');

%% define some parameters

num = length(types);
chunksz=256;
features = 1; % gray = 1, color = 3
imgResizeW = 256;
imgResizeH = 256;

%% create hdf5 file
h5create(hdf5SavePath, '/data', [Inf features imgResizeW imgResizeH], 'Datatype', 'single', 'ChunkSize', [chunksz features imgResizeW imgResizeH]);
h5create(hdf5SavePath, '/label', [Inf 1], 'Datatype', 'single', 'ChunkSize', [chunksz 1]);

%% write image data into the hdf5 file

count = 0;

for i = 1:ceil(num/chunksz)
    data = zeros(chunksz,features,imgResizeW,imgResizeH);
    label = zeros(chunksz,1);
    for j = count + 1:min(chunksz*i,num)
        imPath = [imgPath [filenames{j,1}] '.bmp'];
        im = imread(imPath);
        data(j - count,1,:,:) = imresize(im,[imgResizeW imgResizeH],'bilinear');
        label(j - count,1) = types(j);
    end
    
    if i < ceil(num/chunksz)
        h5write(hdf5SavePath,'/data',single(data),[count+1 1 1 1],[chunksz 1 imgResizeW imgResizeW]);
        h5write(hdf5SavePath,'/label',single(label),[count+1 1],[chunksz 1]);
    else
        h5write(hdf5SavePath,'/data',single(data(1:num-count,:,:,:)),[count+1 1 1 1],[num-count 1 imgResizeW imgResizeW]);
        h5write(hdf5SavePath,'/label',single(label(1:num-count,1)),[count+1 1],[num-count 1]);
    end
    count = min(chunksz*i,num);
    disp('Processing...')
    disp(count)
end