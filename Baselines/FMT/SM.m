function rankList = SM(data)
%SM Summary of this function goes here: Implement [1] (Yu et al's paper) 
%   Detailed explanation goes here
% INPUTS:
%   (1) data - a n_sampels*(n_features+1) matrix, the last column is the
%   label belonging to {0,1} where 1 denotes the positive class (i.e., the minority)
% OUTPUTS:
%   (1) rankList - a 1*n_features vector which is made of the index of features and sorted by their importance in a
%   descending order.
%
% Reference: [1] Yu, Qiao, et al. "A feature selection approach based on a similarity measure for software defect prediction." 
%            Frontiers of Information Technology & Electronic Engineering 18.11 (2017): 1744-1753.


%% Z-score
[temp,mu,std]=zscore(data(:,1:end-1));
data(:,1:end-1) = temp;

dataXPos = data(data(:,end)~=0,1:end-1); % defective instances 
dataXNeg = data(data(:,end)==0,1:end-1); % non-defective instances 
d = size(data,2)-1;
n1 = size(dataXPos,1);
n2 = size(dataXNeg,1);
k = floor(n2/n1); % Used by Yu et al., see line 5 Algo.1 in [1]

if k>=1
    D = pdist2(dataXPos, dataXNeg); % Euclidean distance between defective instance and non-defective instance
    D = D - eye(size(D,1),size(D,1)); % 
    [~, idx] = sort(D, 2); % sort each row of D in ascending order
    idx = idx(:,2:end); 
    
    w = zeros(1,d); %
    temp = zeros(1,d);
    for i=1:n1
        for j=1:k
            difference = abs(dataXPos(i,:)-dataXNeg(j,:)); % Calculate the difference
            [~, idx] = sort(difference); 
            
            for j0=1:d
                temp(j0) = find(idx==j0);
            end
            w = w + temp;
        end
    end
    [~, rankList] = sort(w, 'descend'); 
else
    rankList = 1:d; 
end

end

