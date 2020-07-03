function [Cn,Xn]=centroid_align(x,str)
% Align the original data covariances by congruent transform
% Input:
%   x: the original data covariances K*T*N
%   str: congruent transform by Riemanian or Euclidean mean
% Output:
%   Cn: centralized covariance matrices K*K*N
%   Xn: centralized raw data K*T*N

% Reference:
%   Wen Zhang, Dongrui Wu, “Manifold Embedded Knowledge Transfer for Brain-Computer
%   Interfaces,” IEEE Trans. on Neural Systems & Rehabilitation Engineering, 28(5), pp. 1117-1127, 2020.

tmp_cov=zeros(size(x,1),size(x,1),size(x,3));
for i=1:size(x,3)
    tmp_cov(:,:,i)=cov(x(:,:,i)');
end

C = mean_covariances(tmp_cov,str);
P = C^(-1/2);

Cn=zeros(size(x,1),size(x,1),size(x,3));
for j=1:size(x,3)
    Cn(:,:,j)=P*squeeze(tmp_cov(:,:,j))*P;
end

Xn=zeros(size(x));
for i=1:size(x,3)
    Xn(:,:,i)=P*x(:,:,i);
end
