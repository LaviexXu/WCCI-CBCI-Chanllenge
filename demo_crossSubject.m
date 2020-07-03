% Clinical BCI Challenge-WCCI 2020 Prediction
% Author: Wen Zhang, Yifan Xu
% Date: Jul. 1, 2020
% E-mail: wenz@hust.edu.cn

clear all; close all; warning off;

%% Load datasets
% 8 subjects, each 80*12*(512*5) (trails*channels*points)
root='./data/';
listing = dir([root '*T.mat']);
addpath('./utils/');


%% Load data and perform centroid alignment
delay = [0,0.25,0.5,0.75,1];
samp_len = [2,3,4,5];
de = 3; sl = 2; % the best para
fnum =length(listing);
Ca = cell(fnum,1);
Ya = cell(fnum,1);
ref = {'riemann','logeuclid','euclid'};
for s=1:fnum
    load([root listing(s).name])
    data_proc = preprocess(RawEEGData, cueAt, sampRate, delay(de), samp_len(sl), 4, 40);
    x = nan(12,512*samp_len(sl),80); % reshape to channels*points*trails
    for i=1:80; x(:,:,i)=squeeze(data_proc(i,:,:)); end
    Ya{s} = Labels;
    Ca{s} = centroid_align(x,ref{2});
end


%% Leave one subject out validation
BCA = zeros(1,fnum);
kappa = zeros(1,fnum);
for s = 1:fnum

    % Single target data & multi source data
    ids = 1:fnum; ids(s) = [];
    Yt=Ya{s}; Ys=cat(1,Ya{ids});
    Ct=Ca{s}; Cs=cat(3,Ca{ids});

    % Logarithmic mapping on aligned covariance matrices
    Xs = logmap(Cs,'MI'); % dimension: 78*560 (features*samples)
    Xt = logmap(Ct,'MI');

    % Selective Pseudo-Labeling (SPL)
    options.dim = 10; % subspace dimension
    options.iter = 10; % iteration
    options.alpha = 1; % regularization
    Ypre = SPL(Xs',Ys,Xt',options);

    % Results
    BCA(s) = mean(Ypre==Yt);
    Ypre = reshape(Ypre,1,size(Ypre,1));
    Yte = reshape(Yt,1,size(Yt,1));
    [~,cm,~,~] = confusion(Yte-1,Ypre-1);
    kappa(s) = kappaIndex(cm);
end
fprintf('mean kappa: %.4f\n',mean(kappa));
fprintf('mean BCA: %.4f\n',mean(BCA));


%% Evaluate on the unseen target data
Ys = cat(1,Ya{:}); Cs = cat(3,Ca{:}); Xs = logmap(Cs,'MI');
eval = {'parsed_P09E.mat','parsed_P10E.mat'}; Ytest = [];
for s = 1:2
    load(['./data/' eval{1}])
    data_proc = preprocess(RawEEGData, cueAt, sampRate, delay(de), samp_len(sl), 4, 40);
    x = nan(12,512*samp_len(sl),40);
    for i=1:40; x(:,:,i)=squeeze(data_proc(i,:,:)); end
    Ct = centroid_align(x,ref{2});
    Xt = logmap(Ct,'MI');
    options.dim = 10;
    options.iter = 10;
    options.alpha = 1;
    Ypre = SPL(Xs',Ys,Xt',options);
    Ytest = [Ytest,Ypre];
end
rmpath('./utils/');


%% Save to xlsx
filename = 'HUSTBCI_yfx_CrossSubject.xlsx';
sheet = ['P09', 'P10'];
for j=1:2
    Ypred = Ytest(:,j);
    TrialIdx = 1:40;
    data = table(TrialIdx',Ypred);
    data.Properties.VariableNames = {'Subject Name','Trial index','Prediction (class1=1, class2=2)'};
    writetable(data,filename,'Sheet',sheet(j));
end


