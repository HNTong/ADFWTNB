function new_train = removeNoise(train)
% REMOVENOISE Summary of this function goes here: Handling noise
%   Detailed explanation goes here
% INPUTS:
%   (1) train : a n*(d+1) matrix where the last column is the label which
%   belong to {0,1} 0 - nondefective, 1 - defective.
%   (2) distType: 'euclidean' or 'edr'. 
%   (3) N: Top N nearest instances having different label. 
%   (4) sigma: a given threshold [0,1].
%   (5) epsilon: a given threshold [0,1].
% OUTPUTS:
%  new_train: 
%
% Reference:S. Wang, T. Liu, J. Nam and L. Tan, "Deep Semantic Feature
% Learning for Software Defect Prediction," in IEEE Transactions on
% Software Engineering, vol. 46, no. 12, pp. 1267-1293, 1 Dec. 2020, doi:
% 10.1109/TSE.2018.2877612.

temp = CLNI(train);
new_train = filterInfrequent(temp);

end


function new_train = CLNI(train, distType, N, sigma, epsilon)
%CLNI Summary of this function goes here: Closest List Noise Identification to Modify Noisy data Label.
%   Detailed explanation goes here
% INPUTS:
%   (1) train : a n*(d+1) matrix where the last column is the label which
%   belong to {0,1} 0 - nondefective, 1 - defective.
%   (2) distType: 'euclidean' or 'edr'. 
%   (3) N: Top N nearest instances having different label. 
%   (4) sigma: a given threshold [0,1].
%   (5) epsilon: a given threshold [0,1].
% OUTPUTS:
%  new_train: 
%
% Reference:Sunghun Kim, Hongyu Zhang, Rongxin Wu, and Liang Gong. 2011.
%   Dealing with noise in defect prediction. In Proceedings of the 33rd
%   International Conference on Software Engineering (ICSE '11). Association
%   for Computing Machinery, New York, NY, USA, 481¨C490.
%   DOI:https://doi.org/10.1145/1985793.1985859.
% 

%% Default value
if all(unique(train(:,end))==[0,1])
    error('Label must belong to {0,1}');
end

if ~exist('N','var')||isempty(N)
    N = 5;
end
if ~exist('sigma','var')||isempty(sigma)
    sigma = 0.6;
end
if ~exist('epsilon','var')||isempty(epsilon)
    epsilon = 0.99;
end
if ~exist('distType','var')||isempty(distType)
    distType = 'edr'; % 'edr'
end

%%
A1 = [];
A2 = [];
tol = 0.1; % 
for j = 1:10 % each iteration
    for i = 1:size(train, 1) % each instance
        p = 1;
        myDist = zeros(size(train, 1) - 1, 1);
        for k = 1:size(train, 1)  
            if k~=i
                if ismember(k, A1)
                    continue;
                else
                    if strcmp(distType,'euclidean')
                        myDist(p) = sqrt(sum((train(i) - train(k)).^2));
                    elseif strcmp(distType,'edr')
                        myDist(p) = edr(train(i), train(k), tol);
                    end
                    p = p + 1;
                end
            end    
        end
        
        [val, idx] = topkrows(myDist, N, 'ascend');
        theta = sum(train(idx,end)~=train(i,end)) / N;
        
        if theta >= sigma
            A2 = [A2, i];
        end
    end % end of each instance
    
    if numel(A2)>0 && numel(intersect(A2, A1)) / max(numel(A1), numel(A2)) >= epsilon
        break;
    end
    A1 = A2;
end

new_train = train;
for i=1:numel(A2)
    if new_train(A2(i), end) == 1
        new_train(A2(i), end) = 0;
    else
        new_train(A2(i), end) = 1;
    end
end

end


function new_train = filterInfrequent(train, threshold)
%FILTERFREQUENT Summary of this function goes here: Filter out (Number to Zero) the token
% if its number of occurrences is less than three.
%   Detailed explanation goes here
% INPUTS:
%   (1) train:
%   (2) threshold:
% OUTPUTS:
%   new_train: 
%

if ~exist('threshold','var')||isempty(threshold)
    threshold = 3;
end
trainX = train(:,1:end-1);
val = unique(trainX);
for i=1:length(val)
    if length(find(trainX==val(i))) < threshold
        trainX(find(trainX==val(i))) = 0;
    end
end
new_train = [trainX, train(:,end)];

end