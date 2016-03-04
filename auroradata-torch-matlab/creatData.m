clear all

%% get all images' absolute path and corresponding type

trainLPath = '/home/ljm/AuroraImgData_gray/Exp1_train/Exp1_aurora_train.text';
trainDPath = '/home/ljm/AuroraImgData_gray/Exp1_train/';
testLPath = '/home/ljm/AuroraImgData_gray/Exp1_aurora_val.text';
testDPath = '/home/ljm/AuroraImgData_gray/Exp1_val/';

[filenames_tr type_tr] = textread(trainLPath,'%s%d');
[filenames_te type_te] = textread(testLPath,'%s%d');

%% define some parameters

trainNum = length(type_tr);
testNum = length(type_te);
chunksz=256;
features = 1; % gray = 1, color = 3
imgResizeW = 256;
imgResizeH = 256;

%% create hdf5 files

h5create('traindata.h5', '/data', [Inf features imgResizeW imgResizeH], 'Datatype', 'single', 'ChunkSize', [chunksz features imgResizeW imgResizeH]);
h5create('traindata.h5', '/label', [Inf 1], 'Datatype', 'single', 'ChunkSize', [chunksz 1]);
h5create('testdata.h5', '/data', [Inf features imgResizeW imgResizeH], 'Datatype', 'single', 'ChunkSize', [chunksz features imgResizeW imgResizeH]);
h5create('testdata.h5', '/label', [Inf 1], 'Datatype', 'single', 'ChunkSize', [chunksz 1]);

%% write aurora data into hdf5 files

% train data
count = 0;
% shuflle
sh = randperm(trainNum);

for i = 1:ceil(trainNum/chunksz)
    data = zeros(chunksz,features,imgResizeW,imgResizeH);
    label = zeros(chunksz,1);
    for j = count + 1:min(chunksz*i,trainNum)
        imPath = [trainDPath [filenames_tr{sh(j),1}]];
        im = imread(imPath);
        data(j - count,1,:,:) = imresize(im,[imgResizeW imgResizeH],'bilinear');
        label(j - count,1) = type_tr(sh(j));
    end
    
    if i < ceil(trainNum/chunksz)
        h5write('traindata.h5','/data',single(data),[count+1 1 1 1],[chunksz 1 imgResizeW imgResizeW]);
        h5write('traindata.h5','/label',single(label),[count+1 1],[chunksz 1]);
    else
        h5write('traindata.h5','/data',single(data(1:trainNum-count,:,:,:)),[count+1 1 1 1],[trainNum-count 1 imgResizeW imgResizeW]);
        h5write('traindata.h5','/label',single(label(1:trainNum-count,1)),[count+1 1],[trainNum-count 1]);
    end
    count = min(chunksz*i,trainNum);
    disp('train')
    disp(count)
end


% test data
count = 0;
for i = 1:ceil(testNum/chunksz)
    data = zeros(chunksz,features,imgResizeW,imgResizeH);
    label = zeros(chunksz,1);
    for j = count + 1:min(chunksz*i,testNum)
        imPath = [testDPath [filenames_te{j,1}]];
        im = imread(imPath);
        data(j - count,1,:,:) = imresize(im,[imgResizeW imgResizeH],'bilinear');
        label(j - count,1) = type_te(j);
    end
    
    if i < ceil(testNum/chunksz)
        h5write('testdata.h5','/data',single(data),[count+1 1 1 1],[chunksz 1 imgResizeW imgResizeW]);
        h5write('testdata.h5','/label',single(label),[count+1 1],[chunksz 1]);
    else
        h5write('testdata.h5','/data',single(data(1:testNum-count,:,:,:)),[count+1 1 1 1],[testNum-count 1 imgResizeW imgResizeW]);
        h5write('testdata.h5','/label',single(label(1:testNum-count,1)),[count+1 1],[testNum-count 1]);
    end
    count = min(chunksz*i,testNum);
    disp('test')
    disp(count)
end





















