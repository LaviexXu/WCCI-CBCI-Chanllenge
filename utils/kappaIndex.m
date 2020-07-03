function [kappa] = kappaIndex(confusionMatrix)
% compute kappa value from the confusion matrix
a=sum(confusionMatrix, 2);
b=sum(confusionMatrix, 1);
sampleNum = sum(a);
pe=sum(a.*reshape(b,size(confusionMatrix,1),1))/sampleNum^2;
po=sum(diag(confusionMatrix))/sampleNum;
kappa = (po-pe)/(1-pe);
