clear all

imgPath = '/media/ljm/SSD2/Aurora201510/labeled2003_38044_201510/';
imgLabelPath = '/media/ljm/SSD2/Aurora201510/Alllabel2003_38044.txt';
imgSavePath = '/media/ljm/Data/All_38044_JPG_lossless/';
NUM=4;
for i = 1:NUM
    if ~exist([imgSavePath num2str(i)],'file')
        dirname = [imgSavePath num2str(i)];
        a = ['mkdir' ' ' dirname];
        system(a);
    else
        dirname = [imgSavePath num2str(i)];
        disp ([dirname ' ' 'has existed!']);
    end
end

[label,type] = textread(imgLabelPath,'%s%d');

fileFoundNum = 0;
fileNotFoundNum = 0;
typeNUM = zeros(1,4);
for i = 1:length(label)
    filename = [label{i,1}];
    if exist([imgPath filename '.bmp'],'file')
        disp([imgPath filename '.bmp' ' has been founded!']);
        fileFoundNum = fileFoundNum + 1;
        
        img = imread([imgPath filename '.bmp']);
        [m, n] = size(img);
        imwrite(img, [imgSavePath num2str(type(i)) '/' filename '.jpg'],'JPEG','Quality',100);
        
        typeNUM(type(i)) = typeNUM(type(i)) + 1;
    else
        disp([imgPath '/' filename, '   not found!']);
        fileNotFoundNum = fileNotFoundNum + 1;
    end
end

for j=1:4
    disp(['The total number of type ' num2str(j) ' are ' num2str(typeNUM(j))]);
end

disp(['File found: ' num2str(fileFoundNum)]);
disp(['File not found: ' num2str(fileNotFoundNum)]);