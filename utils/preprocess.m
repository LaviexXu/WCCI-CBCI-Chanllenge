function validData=preprocess(RawEEGData, cueAt, sampRate, delay, sample_length, W1, W2)
% Clinical BCI Challenge-WCCI 2020 Prediction
% Author: Wen Zhang, Yifan Xu
% Date: Jul. 1, 2020
% E-mail: wenz@hust.edu.cn

if nargin<5; sample_length = 3; end
if nargin<6;  W1=4; W2=40; end % default 4-40Hz 

% detrend the signal
detrendData = zeros(size(RawEEGData));
for i = 1:size(RawEEGData,1)
    tmp_data = squeeze(RawEEGData(i, :, :))';
    detrendData(i, :, :) = detrend(tmp_data)';
end

% bandpssing filter
bandPassData = zeros(size(RawEEGData));
Wn1 = [W1*2 W2*2]/sampRate;
[BB,AA] = butter(6,Wn1);
for i=1:(size(detrendData,1))
    data = squeeze(detrendData(i, :, :))';
    data = filter(BB,AA,data);
    bandPassData(i, :, :) = data';
end

% trial select
validData = bandPassData(:, :, ...
            (cueAt+delay)*sampRate+1:(cueAt+delay+sample_length)*sampRate);
