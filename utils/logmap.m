function Fea = logmap(COV,type)
% Logarithmic mapping on centralized signal covariance matrices
% Input:
%   COV: K*K*N, centralized signal covariance matrices
% Output:
%   Fea: tangent space features, d*N

% Reference:
%   Wen Zhang, Dongrui Wu, “Manifold Embedded Knowledge Transfer for Brain-Computer
%   Interfaces,” IEEE Trans. on Neural Systems & Rehabilitation Engineering, 28(5), pp. 1117-1127, 2020.


NTrial = size(COV,3);
N_elec = size(COV,1);

if strcmp(type,'ERP')

    % Select upper right elements related to temporal information
    N = N_elec/2;
    Fea = zeros(N*N,NTrial);
    for i=1:size(COV,3)
        Tn = logm(COV(:,:,i));
        Fea(:,i)=reshape(Tn(1:N,(N+1):end),[],1);
    end 
elseif strcmp(type,'MI')

    % Select upper triangular elements related to spatial information
    Fea = zeros(N_elec*(N_elec+1)/2,NTrial);
    index = reshape(triu(ones(N_elec)),N_elec*N_elec,1)==1;
    for i=1:NTrial
        Tn = logm(COV(:,:,i));
        tmp = reshape(sqrt(2)*triu(Tn,1)+diag(diag(Tn)),N_elec*N_elec,1);
        Fea(:,i) = tmp(index);
    end
end
