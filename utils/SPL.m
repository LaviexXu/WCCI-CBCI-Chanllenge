function Ypre = SPL(Xs,Ys,Xt,options)
    % Selective Pseudo-Labeling (SPL)
    
    % Reference:
    %   Qian Wang, Toby P. Breckon, Unsupervised Domain Adaptation via 
    %   Structured Prediction Based Selective Pseudo-Labeling, In AAAI 2020.
    
    num_class = length(unique(Ys));
    W_all = zeros(size(Xs,1)+size(Xt,1));
    W_s = constructW(Ys);
    W = W_all;
    W(1:size(W_s,1),1:size(W_s,2)) =  W_s;

    for t = 1:options.iter

        P = LPP([Xs;Xt],W,options);
        Ps = Xs*P;
        Pt = Xt*P;
        proj_mean = mean([Ps;Pt]);
        Ps = Ps - repmat(proj_mean,[size(Ps,1) 1 ]);
        Pt = Pt - repmat(proj_mean,[size(Pt,1) 1 ]);
        Ps = L2Norm(Ps);
        Pt = L2Norm(Pt);

        %% distance to class means
        classMeans = zeros(num_class,options.dim);
        for i = 1:num_class
            classMeans(i,:) = mean(Ps(Ys==i,:));
        end
        classMeans = L2Norm(classMeans);
        distClassMeans = EuDist2(Pt,classMeans);
        expMatrix = exp(-distClassMeans);
        probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
        [prob,Ypre] = max(probMatrix,[],2);
        p=1-t/options.iter;
        [sortedProb,index] = sort(prob);
        sortedPredLabels = Ypre(index);

        trustable = zeros(1,length(prob));
        for i = 1:num_class
            prob_class = sortedProb(sortedPredLabels==i);
            if isempty(prob_class)
                trustable = trustable+ (prob>prob_class(floor(length(prob_class)*p)+1)).*(Ypre==i);
            end
        end
        W = constructW([Ys;Ypre]');

        if sum(trustable)>=length(prob)
            break;
        end
    end
end

function W = constructW(label)
    W = zeros(length(label),length(label));
    num_class = max(label(:));
    for i = 1:num_class
        W = W + double(label==i)'*double(label==i);
    end
end

function y = L2Norm(x)
    % x is a feature matrix: one example in a row
    y = x./repmat(sqrt(sum(x.^2,2)),[1 size(x,2)]);
end
