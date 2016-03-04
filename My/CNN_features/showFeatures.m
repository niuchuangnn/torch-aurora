clear all

featuresRoot = '/media/ljm/Data/Experiments/CNNFeatures/All38044/';
configPath = '/media/ljm/Data/Experiments/config/all.text';

[names types] = textread(configPath, '%s%d');
features = zeros(128,length(names));

for i = 1:length(names)
    name = names{i,1}(1:16);
    disp(name) 
    load([featuresRoot name])
    f = reshape(cnn,6,6,128);
    m = mean(mean(f));
    features(:,i) = m(1,1,:);
end