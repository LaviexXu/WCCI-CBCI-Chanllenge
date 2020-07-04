% Clinical BCI Challenge-WCCI 2020 Prediction
% Author: Yifan Xu, Wen Zhang
% Date: Jul. 1, 2020
% E-mail: wenz@hust.edu.cn

load('optimal_param_within.mat');
addpath('./utils/');
Ytest = [];
for sub=1:8
    trainFilePath=strcat('data/parsed_P0', num2str(sub), 'T.mat');    
    testFilePath=strcat('data/parsed_P0', num2str(sub), 'E.mat');    
    trainFile = load(trainFilePath);
    testFile = load(testFilePath);
    trainData = trainFile.RawEEGData;
    trainLabel=trainFile.Labels;
    testData=testFile.RawEEGData;
    
    sample_length = optimal_param{sub}.sample_length;
    delay = optimal_param{sub}.delay;
    if optimal_param{sub}.filter==1
        trainData = trainData(:, :, (3+delay)*512+1:(3+delay+sample_length)*512);
        testData = testData(:, :, (3+delay)*512+1:(3+delay+sample_length)*512);
    else
        trainData = preprocess(trainData, trainFile.cueAt, trainFile.sampRate, delay,sample_length);
        testData = preprocess(testData, testFile.cueAt, testFile.sampRate, delay,sample_length);
    end
    [train_csp, test_csp] = CSPfeature(trainData, trainLabel, testData);
    SVM = fitcsvm(train_csp,trainLabel,'KernelFunction','linear');
    Ypred=predict(SVM,test_csp);
    Ytest = [Ytest,Ypred];
end

%% save to xlsx
filename = 'HUSTBCI_Xu_WithinSubject.xlsx';

for j=1:8
    Ypred = Ytest(:,j);
    TrialIdx = 1:40;
    sub = cell(40,1);
    for i=1:40
        sub{i}=['P0',num2str(j)];
    end
    data = table(sub,TrialIdx',Ypred);
    data.Properties.VariableNames = {'Subject Name','Trial index','Prediction (class1=1, class2=2)'};
    writetable(data,filename,'Sheet',['P0',num2str(j)]);
end

