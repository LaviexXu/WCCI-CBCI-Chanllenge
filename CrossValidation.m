% Clinical BCI Challenge-WCCI 2020 Prediction
% Author: Wen Zhang, Yifan Xu
% Date: Jul. 1, 2020
% E-mail: wenz@hust.edu.cn
clear,clc;
delay=[0,0.25,0.5,0.75,1];
sample_length = [2,3,4];
results=zeros(8,1);
optimal_param=cell(8,1);

for sub=1:8
    file_path=strcat('data/parsed_P0', num2str(sub), 'T.mat');
    load(file_path)
    highest_kappa = 0;
    for filtering=1:2
        for d=1:5
            for s=1:3
    
                if filtering==1
                    preprocessedData = RawEEGData(:, :, (3+delay(d))*512+1:(3+delay(d)+sample_length(s))*512);
                else

                preprocessedData = preprocess(RawEEGData, cueAt, sampRate, delay(d),...,
                                             sample_length(s));    
                end
                pred_y = zeros(1,80);

                for test_sample = 1:80
                    train = 1:80;
                    train(test_sample) = [];
                    trainLabel = Labels(train, :);
                    trainData = preprocessedData(train, :, :);
                    testData = preprocessedData(test_sample, :, :);

                    % extract csp features
                    [train_csp, test_csp] = CSPfeature(trainData,trainLabel,testData);
 
                    % train the LDA model
                    LDA = fitcdiscr(train_csp,trainLabel);
                    pred_y(test_sample)=predict(LDA,test_csp);
                end
                % compute the kappa index in leave-one-out cross validation

                [~,cm,~,~]=confusion(Labels'-1,pred_y-1);
                kappa = kappaIndex(cm);

                if kappa>highest_kappa
                    highest_kappa = kappa;
                    optimal_param{sub}.delay=delay(d);
                    optimal_param{sub}.sample_length=sample_length(s);
                    optimal_param{sub}.filter=filtering;
                end

            end
        end
    end

end
save('within_CSPLDA.mat', 'optimal_param');