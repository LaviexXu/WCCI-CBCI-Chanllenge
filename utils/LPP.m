function P = LPP(data,W,options)
% Locality Preserving Projection

% Reference:
%   He, Xiaofei, Partha Niyogi. "Locality preserving projections." NIPS 2004.

data = double(data);
W = double(W);
D = diag(sum(W,2));
L = D - W;
Sl = data'*L*data;
Sd = data'*D*data;
Sl = (Sl+Sl')/2;
Sd = (Sd+Sd')/2;

Sl = Sl + options.alpha*eye(size(Sl,2));
[P,~] = eigs(double(Sd),double(Sl),options.dim,'la',options);
for i = 1:size(P,2)
    if (P(1,i)<0)
        P(:,i) = P(:,i)*-1;
    end
end
