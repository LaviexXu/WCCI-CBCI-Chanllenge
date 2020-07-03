function [fTrain,fTest]=CSPfeature(xTrain,yTrain,xTest)
%%  train CSP filters
nfilter=3;
nChannel=size(xTrain,2);
cs=unique(yTrain);
xTrain0=xTrain(yTrain==cs(1),:,:);
xTrain1=xTrain(yTrain==cs(2),:,:);
Sigma0=zeros(nChannel);Sigma1=zeros(nChannel);
for i=1:size(xTrain0,1)
    tmp0=cov(squeeze(xTrain0(i, :, :))');
    Sigma0=Sigma0+tmp0;
end
for i=1:size(xTrain1,1)
    tmp1=cov(squeeze(xTrain1(i,:,:))');
    Sigma1=Sigma1+tmp1;
end
Sigma0=Sigma0/size(xTrain0,1);
Sigma1=Sigma1/size(xTrain1,1);
[d,v]=eig(Sigma1\Sigma0);
[~,v_index]=sort(diag(v),'descend');
d_sort=d(:,v_index);
w0=d_sort(:,1:nfilter); %CSP filters
w1=d_sort(:,end-nfilter+1:end); %CSP filters
W=[w0,w1];

fTrain=zeros(size(xTrain,1),size(W,2));
fTest=zeros(size(xTest,1),size(W,2));
for i=1:size(xTrain,1)
    X=W'*squeeze(xTrain(i,:,:));
    fTrain(i,:)=log(diag(X*X')/trace(X*X'));
end
for i=1:size(xTest,1)
    X=W'*squeeze(xTest(i,:,:));
    fTest(i,:)=log(diag(X*X')/trace(X*X'));
end
