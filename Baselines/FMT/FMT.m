function [MCC,F1,AUC] = FMT(source,target, N)
%FMT Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) source - a m*(d1+1) matrix where the last column is the label
%   belonging to {0,1} where 1 denotes the positive class (minority);
%   (2) target - a n*(d2+1) matrix (d2>=d1) where the last column is the label
%   belonging to {0,1} where 1 denotes the positive class (minority);
% OUTPUTS:
%
%
% Reference: Yu, Qiao , S. Jiang , and Y. Zhang . "A feature matching and transfer approach for cross-company defect prediction."
%            Journal of Systems and Software (2017):S0164121217301346.

warning('off');
if ~exist('N','var')||isempty(N)
    N = 200; % Given by Yu et.al. in their paper
end

sourceX = source(:,1:(end-1));
sourceY = source(:,end);

% m = size(source, 1);
n = size(target, 1);

%% Feature selection

% mat to ARFF
label = cell(size(source,1),1);
for i=1:size(source,1)
    if source(i,end)==0
        label{i} = 'No';
    else
        label{i} = 'Yes';
    end
end
feaClaNames = cell(1, size(source,2));
for j0=1:(size(source,2)-1)
    feaClaNames{j0} = ['X',num2str(j0)];
end
feaClaNames{end} = 'Defect';
sourceARFF = matlab2weka('data', feaClaNames,  [num2cell(source(:,1:end-1)), label]);

% perform CFS 
cfs = javaObject('weka.attributeSelection.CfsSubsetEval');
% javaaddpath('consistencySubsetEval.jar'); 
% confs = javaObject('weka.attributeSelection.ConsistencySubsetEval'); %
attrSelector = javaObject('weka.attributeSelection.AttributeSelection');
searchMethod = weka.attributeSelection.BestFirst();
% filter.setOptions(weka.core.Utils.splitOptions(['-R ',num2str(insts.numAttributes())]));
% filter.setInputFormat(insts);
attrSelector.setEvaluator(cfs);
attrSelector.setSearch(searchMethod);
attrSelector.SelectAttributes(sourceARFF);
selAttrs = attrSelector.selectedAttributes(); 
% a=attrSelector.toResultsString();
sourceCFS = source(:,selAttrs+1);

m = size(sourceCFS,2)-1;
if m<=n
    source = sourceCFS;
else
    rankList = SM(source); % Call SM algorithm
    source = source(:,[rankList(1:min(n,length(rankList))), end]); 
end

%% Feature matching
d1 = size(source, 2)-1;
d2 = size(target, 2)-1;

% the number of sample may be smaller than N
minimum = min(size(source,1), size(target,1));
if minimum<N
    N = minimum;
end

rand('seed',0); 
idx1 = randperm(size(source,1), N);
idx2 = randperm(size(target,1), N);
subSrc = source(idx1,:);
subTar = target(idx2,:);

% Yu et al. said that strategy1 is better or comparable to strategy2, so
% strategy1 is used here.
matchFea = zeros(1,d1);
for i=1:d1
    for j=1:d2
        feaSrc = sort(subSrc(:,i)); % ascending order
        feaTar = sort(subTar(:,j));
        
        cumFeaSrc = cumsum(feaSrc);
        cumFeaTar = cumsum(feaTar);
        
        temp = 0;
        for k=1:N-1
            areaSrc_k = (cumFeaSrc(k)+cumFeaSrc(k+1))*1/2;
            areaTar_k = (cumFeaTar(k)+cumFeaTar(k+1))*1/2;
            temp = temp + abs(areaSrc_k-areaTar_k);
        end
        D(i,j) = temp;
    end
    minimum = min(D(i,:));
    idx = find(D(i,:)==minimum(1));
    matchFea(i) = idx(1);
end

target = target(:,[matchFea, end]);

%% Training on the source data with NaiveBayesB
label = cell(size(source,1),1);
for i=1:size(source,1)
    if source(i,end)==0
        label{i} = 'No';
    else
        label{i} = 'Yes';
    end
end
feaClaNames = cell(1, size(source,2));
for j0=1:(size(source,2)-1)
    feaClaNames{j0} = ['X',num2str(j0)];
end
feaClaNames{end} = 'Defect';
sourceARFF = matlab2weka('data', feaClaNames,  [num2cell(source(:,1:end-1)), label]);
sourceARFF.setClassIndex(sourceARFF.numAttributes()-1);
model = javaObject('weka.classifiers.bayes.NaiveBayes');
model.buildClassifier(sourceARFF);

%% Prediction on the target data
labelTar = cell(size(target,1),1);
for i=1:size(target,1)
    if target(i,end)==0
        labelTar{i} = 'No';
    else
        labelTar{i} = 'Yes';
    end
end
targetARFF = matlab2weka('data', feaClaNames,  [num2cell(target(:,1:end-1)), labelTar]);
targetARFF.setClassIndex(targetARFF.numAttributes()-1);

% pred = zeros(m,1);
classProbsPred = zeros(size(target,1),2); % 2 - two classes
for i = 0:(size(target,1)-1)
%     pred(i+1) = model.classifyInstance(train.instance(i));
    classProbsPred(i+1,:) = model.distributionForInstance(targetARFF.instance(i));
end
probPos = classProbsPred(:,2); % the probability of being positive

try
    [MCC,F1,AUC] = Performance(target(:,end), probPos); % Call Performance()
catch
    AUC=nan;MCC=nan;F1=nan;
end

end

