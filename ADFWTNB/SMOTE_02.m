function balancedData = SMOTE_02(data, final_ratio, k, rand_state)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) data - unbalanced dataset, a T*(d+1) matrix, d denotes the number of features, the last column is the label, 0/1 - nondefective/defective.
%   (2) k (default 5)   - the number of nearest neighbors;
%   (3) final_ratio (1, default) - the ideal ratio of the number of positive samples to that of negative samples;
%   (4) rand_state (0, default)  - random seed;
% OUTPUTS:
%   balancedData.
%
% Reference: N.V. Chawla, K.W. Bowyer, L.O. Hall, and W.P. Kegelmeyer, SMOTE:
% synthetic minority over-sampling technique, Journal of Artificial Intelligence 
% Research, vol.16, pp.321¨C357, 2002


%% Default value
if ~exist('final_ratio', 'var')||isempty(final_ratio)
    final_ratio = 1; % 
end

if ~exist('rand_state','var')||isempty(rand_state)
    rand_state = 1;
end
if ~exist('n_neighbor','var')||isempty(k)
    k = 5;
end

%% 
if final_ratio<=(sum(data(:,end)==1)/sum(data(:,end)==0)) % 
    balancedData = data; % Do not need resampling on defective modules
    return;
end

numPos = sum(data(:,end)~=0); % Number of defectvie samples
dataY = data(:,end);          % Label of the unbalanced dataset
neg_size = sum(dataY==0);
pos_size = sum(dataY~=0);


N = round((neg_size*final_ratio-pos_size)/pos_size)*100;
if N<0 %
    balancedData = data;
    return;
end

if pos_size*(1+N/100)>neg_size 
    N = N - 100; 
    if N==0
        balancedData = data;
        return;
    end
end

if N <100
    numPos = floor((N/100)*numPos);
    N = 100;
end

% if nargin==3
%     k = 5; % Number of the nearest neighbors.
% end

% shuffle the minority samples
dataPos = data(data(:,end)~=0,:);
rand('seed', rand_state);
idx = randperm(numPos, numPos); % Disturb the order of minority samples
dataPos = dataPos(idx,:);

N = floor(N/100);
numattrs = size(dataPos,2) - 1;

synData = zeros(N*numPos,size(dataPos,2)-1);

dataX = dataPos(:,1:end-1);

% Distance between instances
distM = dist(dataX');
distM = distM - eye(size(distM,1),size(distM,1)); 

% Index of neighbors of T instance
% idx = randperm(size(data,1),T); % Disturb the order of minority samples
neigIndex = zeros(numPos, k);    % Initialization 
for i=1:numPos % each minority sample
    [val, ord] = sort(distM(i,:)); % smallest->biggest for each row
    neigIndex(i,:) = ord(2:(k+1));
end

count = 1;
while N~=0
    for i = 1:numPos % each minority samples
        sample = dataX(i,:);
        nn = randperm(k,1);
        synData(count,:) = sample + rand*(dataX(neigIndex(i,nn),:)-sample);
        count = count + 1;
    end
    N = N-1;
end

synData = [synData, ones(size(synData,1),1)];

balancedData = [data; synData];
end

