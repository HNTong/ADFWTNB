function [MCC, F1, AUC] = DFWTNB(Source,Target,isDiscrete,sigma)
%DFWTNB Summary of this function goes here
%   Detailed explanation goes here:
%
% INPUTS:
%   (1) Source - each row of Source corresponds to a observation and the last column is the label belonging to {0,1}.
%   (2) Target - test data from target project; each row is a observation. Source and TargetData have same metrics, namely the number of columns is equal.
%   (3) isDistance - {0,1} - 1 denotes that each metric is  discrete, otherwise continuous.
%   (4) sigma - 
% OUTPUTS:
%   Performance measures
%

warning('off');
% Default value
if nargin==2
    isDiscrete = 0;
    sigma = 1;
end


% Perform oversampling
Source = SMOTE_02(Source,1); 


if size(Source,2)~=size(Target,2)
    error('Datastes must have same number of metrics!')
end


% Transform element 0 into a very small positive number  
for i = 1:size(Target,2)-1 % Each feature
	% Target
    if max(Target(:,i)) == min(Target(:,i))&&max(Target(:,i)) == 0
        Target(:,i) = repmat(0.00001,size(Target,1),1);
    end
	% Source
    if max(Source(:,i)) == min(Source(:,i))&&max(Source(:,i)) == 0
        Source(:,i) = repmat(0.00001,size(Source,1),1);
    end
end


% 0 -> 1e-4 for the following logorithm
epsilon = 1e-4;
temp = Source(:,1:end-1);
temp(find(temp==0))=epsilon;
Source(:,1:end-1) = temp;

temp = Target(:,1:end-1);
temp(find(temp==0))=epsilon;
Target(:,1:end-1) = temp;


% Take logorithm (do this only all values are positive) - either log(x) or log10(x), here we use natural logrithm.
Source(:,1:end-1)=log(Source(:,1:end-1)); 
Target(:,1:end-1)=log(Target(:,1:end-1));


% Maximal Information Coefficient (MIC) (1) the range of values is [0,1]; (2) the bigger the better; 
% (3) Reshef, David N., et al. Detecting novel associations in large data sets. science 334.6062 (2011): 1518-1524. 
myMIC = [];
for i=1:(size(Source,2)-1) % Each metric
    minestats = mine(Source(:,i)',Source(:,end)'); % input must be row vector
    myMIC(i) = minestats.mic; %[0,1]
end

% % Normalization
myMIC = myMIC / sum(myMIC); 


% maximun and minimum of each feature in target
Max = max(Target(:,1:end-1),[],1);% the maximum value of each feature
Min = min(Target(:,1:end-1),[],1);

% Calculate similarity and weight of each training instance based on target
s = zeros(size(Source,1),1);% s - the similarity of each training instance
w = zeros(size(Source,1),1);% w - the weight of each training instance
for i=1:size(Source,1) % each source instance
    
    % Weighting
    tem=0;
    for j=1:size(Max,2) % each feature
        if Source(i,j)>=Min(1,j)&&Source(i,j)<=Max(1,j)
            tem=tem+1*myMIC(j);
        end
    end    
    s(i,1)=tem;
    w(i,1)=s(i,1)/(sum(myMIC)-s(i,1)+1)^2;%
    
end


% Calculate the prior probability of each classes (i.e., positive class and negative class)
label = Source(:,end);                     % the label of source
num_pos = length(find(label==1));          % the number of  positive instances
num_neg = length(find(label==min(label))); % the number of  negative instances
pri_prob_pos = (sum(w(find(label==1))) + 1) / (sum(w) + numel(unique(label)));
pri_prob_neg = (sum(w(find(label==min(label)))) + 1) / (sum(w) + numel(unique(label)));

pred_label = zeros(size(Target,1),1);
pro_pos    = zeros(size(Target,1),1);

for i=1:size(Target,1) % Each instance in Target   
    met=Target(i,1:end-1);% i-th instance in Target
    walls=[];
    idx1=[];
    idx2=[];
    n=0;
    pos_cond_met = [];
    neg_cond_met = [];
    for j=1:length(met) % Each feature
        if isDiscrete % 
            n=numel(unique(Source(:,j))); % the total number of unique values of j-th metric in Source
            idx1 = find((Source(:,j)==met(j))&(label==1));          % the total number of positive instances in source whose j-th metric value equals to met(j)
            idx2 = find((Source(:,j)==met(j))&(label==min(label))); % the total number of negative instances in source whose j-th metric value equals to met(j)
        else % 
            walls = fayyadIrani(Source(:,j),label); % Call fayyadIrani() to generate walls.
            walls = sort([min(Source(:,j)) + min(Source(:,j))/2,walls,max(Source(:,j))]); %  
            n = length(walls)-1; % the number of intervals    
            
            % [infimum, supremum] is the most nearest wall interval which includes met(j) in ideal condition.
            supremum = walls(min(find(roundn(walls,-6)>=roundn(met(j),-6))));% find supremum of j-th metric in walls.
            infimum = walls(max(find(roundn(walls,-6)<=roundn(met(j),-6)))); % find infimum of j-th metric in walls. If infimum is empty,... 
            
             % To void supremum or infimum is empty when met(j) is larger than max(walls) or met(j) is smaller than min(walls).
            if isempty(supremum)
                supremum = max(walls);
            end
            if isempty(infimum)
                infimum = min(walls);
            end
            
            idx1 = find((Source(:,j)>infimum)&(Source(:,j)<=supremum)&(label==1)); % the total number of positive instances which belong to the interval (infinum,supremum].
            idx2 = find((Source(:,j)>infimum)&(Source(:,j)<=supremum)&(label==min(label)));         
        end
        
        % Calculate the class conditional probability
        pos_cond_met(j) = (sum(w(idx1)) + 1) / (sum(w(find(label==1))) + n);          % Calculate the class-conditionnal peobability for j-th metric   
        neg_cond_met(j) = (sum(w(idx2)) + 1) / (sum(w(find(label==min(label)))) + n); % Calculate the class-conditionnal peobability for j-th metric 
    end
    
    
    % Calculate posterior probability
    deno = pri_prob_pos * prod(power(pos_cond_met,exp(-myMIC/sigma^2))) + pri_prob_neg * prod(power(neg_cond_met,exp(-myMIC/sigma^2))); % prod([1,2,3])=>6
    pro_pos(i)=pri_prob_pos * prod(power(pos_cond_met,exp(-myMIC/sigma^2))) / deno;
    pro_neg(i)=pri_prob_neg * prod(power(neg_cond_met,exp(-myMIC/sigma^2))) / deno;  

    % Predicted labels
    pred_label(i) = double(pro_pos(i)>=pro_neg(i));
    
end

try
    [MCC, F1, AUC ] = Performance(Target(:,end), pro_pos); % Call self-defined Performance()
catch
    F1=nan;AUC=nan;MCC=nan;
end

