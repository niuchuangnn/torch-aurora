function setPathes(kkk)
	trainConfig = configPathRoot .. configNames_train[kkk]
	testConfig = configPathRoot .. configNames_test[kkk]
    modelpath = modelSaveRoot .. 'model' .. string.sub(configNames_train[kkk], 6, 10) .. '.bin'
	mean_std_path = mean_std_root .. 'mean_std' .. string.sub(configNames_train[kkk], 6, 10) .. '.bin'
end
