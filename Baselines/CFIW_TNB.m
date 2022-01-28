function [MCC,F1,AUC] = CFIW_TNB(source, target, k)
%CFIW-TNB Summary of this function goes here: 
%   Detailed explanation goes here
% INPUTS:
%   (1) source - a n1*(d+1) matrix, the last column is the label where 0/1
%   denotes the non-defective/defective module
%   (2) target - a n2*(d+1) matrix, the last column is the label where 0/1
%   denotes the non-defective/defective module 
%   (3) k      - Number of neighbors
% 
% 
% Reference:[1] Zou, Quanyi & Lu, Lu & Qiu, Shaojian & Gu, Xiaowei & Cai,
% Ziyi. (2021). Correlation feature and instance weights transfer learning
% for cross project software defect prediction. IET Software. 15.
% 10.1049/sfw2.12012.
% [2]Jiang, L., et al: A correlation based feature weighting flter for naive
%Bayes. IEEE Trans. Knowl. Data Eng. 31(2), 201¨C213 (2019).


assert(size(source,2)==size(target,2), 'Two datastes must have the same number of features.')

warning('off');

if ~exist('k','var')||isempty(k)
    k = ceil(0.02 * size(target,1)); % see Section 4.4 in [1]
end

%% Part 1: determine instance weights
n1 = size(source,1);  
s = zeros(n1,1);      % Initialize the similarity
m = size(source,2)-1; % number of features in source data
wIns = zeros(n1,1);   % Initialize the weight of source instance
maxFeaTar = max(target(:,1:end-1));
minFeaTar = min(target(:,1:end-1));
for i=1:size(source, 1) % each source instance
    insSrcX = source(i,1:end-1);
    [~, idxNearNeib] = pdist2(target(:,1:end-1), insSrcX, 'euclidean', 'Smallest', k); % k smallest target neighbors for i-th source instance
    nearNeibX = target(idxNearNeib, 1:end-1);
    insRep = repmat(insSrcX,k,1);      % Copy current source instance k times and create a k*m matrix
    temp = (insRep - nearNeibX) .* double((insRep>=repmat(minFeaTar,k,1))&(insRep<=repmat(maxFeaTar,k,1)));
    s(i) = sum(1./(1+sum(abs(temp)))); % see Eqs.(10) and (11) in [1] 
    wIns(i) = s(i)/(m-s(i)+1)^2;       % see Eq.(12) in [1]
end

%% Part2: determine feature weights
javaaddpath('D:\Program Files\Weka-3-8-4\weka.jar');
wekaSrc = mat2ARFF(source, 'classification'); % self-defined function 'mat2ARFF'
wekaTar = mat2ARFF(target, 'classification');

% Discretize source dataset by using Fayyad & Irani's MDL method
filterDisc = javaObject('weka.filters.supervised.attribute.Discretize');  % weka.attributeSelection.InfoGainAttributeEval
filterDisc.setInputFormat(wekaSrc);
filterDisc.setOptions(weka.core.Utils.splitOptions('-R first-last')); % all attributes
wekaSrcDisc = weka.filters.Filter.useFilter(wekaSrc, filterDisc); 

% Discretize target dataset by using unsupervised discretization method
filterDiscTar = javaObject('weka.filters.unsupervised.attribute.Discretize'); % Do not use class label for target dataset
filterDiscTar.setInputFormat(wekaTar);
filterDiscTar.setOptions(weka.core.Utils.splitOptions(['-R 1-20', ' -B 20'])); % all attributes
wekaTarDisc = weka.filters.Filter.useFilter(wekaTar, filterDiscTar); 

[matSrc,featureNames,targetNDX,stringVals,relationName1] = weka2matlab(wekaSrcDisc,[]); % norminal to number (starting from 0)


filterIG = javaObject('weka.attributeSelection.InfoGainAttributeEval');
attrSelector = javaObject('weka.attributeSelection.AttributeSelection');
attrSelector.setEvaluator(filterIG);
searchMethod = weka.attributeSelection.Ranker(); 
attrSelector.setSearch(searchMethod); 
attrSelector.SelectAttributes(wekaSrcDisc);

feaClassInfGain = zeros(m, 1); % Initialize the information gain (i.e., mutual information) between each fature and the class label for source data
feaFeaInfGain = zeros(m, m);   % Initialize the information gain each pair of features in source data
KLDivSrcTar = zeros(m,1);      % Initialize the KL divergence between each source feature and the corresponding target feature
deleIdx = [];                  % Initialize the index of feaures which have only one unique value
for i=1:m  % each feature
    
    % feature class relevance
    feaClassInfGain(i) = filterIG.evaluateAttribute(i-1);
    
    % KL divergence between source and target features
    attrStas = wekaSrcDisc.attributeStats(i-1);  % Statistics of current feature for source data
    disCount = double(attrStas.distinctCount()); % number of different values
    nomiNum = double(attrStas.nominalCounts());  % a column vector demotes the number of instances of each unique value
    vec1 = nomiNum/(sum(nomiNum));
    if disCount == 1
        deleIdx = [deleIdx, i];
    end
    
    
    filterDiscTar = javaObject('weka.filters.unsupervised.attribute.Discretize'); % Do not use class label for target dataset
    filterDiscTar.setInputFormat(wekaTar);
    filterDiscTar.setOptions(weka.core.Utils.splitOptions(['-R ', num2str(i), ' -B ', num2str(disCount)])); 
    wekaTarDisc = weka.filters.Filter.useFilter(wekaTar, filterDiscTar);
    wekaSrcDisc.setRelationName('target');
    wekaTar = wekaTarDisc;
    attrStasTar = wekaTarDisc.attributeStats(i-1); % Statistics of current feature for target data
    disCount = attrStasTar.distinctCount();        % Number of different values
    nomiNumTar = double(attrStasTar.nominalCounts()); % A column vector demotes the number of instances of each value
    vec2 = nomiNumTar/(sum(nomiNumTar));              % Probability of each unique value
    KLDivSrcTar(i) = KLDiv(vec1, vec2);         % Self-defined function
%     KLDivSrcTar(i) = JSDivergence(vec1, vec2);
    % 
end
[matTar,featureNames,targetNDX,stringVals,relationName1] = weka2matlab(wekaTarDisc,[]);

% Remove features having only one kind of value, otherwise an error will be arised.
wekaSrcDiscCopy = wekaSrcDisc;
for i=numel(deleIdx):-1:1
    wekaSrcDiscCopy.deleteAttributeAt(int32(deleIdx(i)-1));
end

for i=1:(wekaSrcDiscCopy.numAttributes-1) %each feature in source data
    wekaSrcDiscCopy.setClassIndex(i-1)
    filterIGSrc = javaObject('weka.attributeSelection.InfoGainAttributeEval');
    filterDisc.setInputFormat(wekaSrcDiscCopy);
    attrSelector = javaObject('weka.attributeSelection.AttributeSelection');
    attrSelector.setEvaluator(filterIGSrc);
    searchMethod = weka.attributeSelection.Ranker();
    attrSelector.setSearch(searchMethod);
    attrSelector.SelectAttributes(wekaSrcDiscCopy);
    for j=i:(wekaSrcDiscCopy.numAttributes-1)
        str = string(wekaSrcDiscCopy.attribute(j-1)); % variable name
        temp = split(str, ' ');
        if j~=i
            feaFeaInfGain(i, str2double(strrep(temp(2), 'X',''))) = filterIGSrc.evaluateAttribute(j-1);
        end 
    end
end

% Normalization
KLDivSrcTar = KLDivSrcTar/sum(KLDivSrcTar);
feaFeaInfGain = feaFeaInfGain + feaFeaInfGain';
feaFeaInfGain = feaFeaInfGain/((1/m*(m-1))*sum(feaFeaInfGain,'all')); 
feaClassInfGain = feaClassInfGain/sum(feaClassInfGain);

D = feaClassInfGain - KLDivSrcTar - 1/(m-1)*sum(feaFeaInfGain, 2); % see Eq.(16) in [1]

wFea = 1./(1+exp(-D)); % weight of feature, see Eq.(20) in [1]

%% Prior probability based on training data
uniValue = unique(source(:,end)); % a column vector sorted ib in ascending order
q = numel(uniValue); %  
priorProb = zeros(1, q); % [negative, positive]
for i = 1:q
    priorProb(i) = (1+sum(wIns(matSrc(:,end)==uniValue(i))))/(sum(wIns)+q); % see Eq.(22) in [1]
end


%% Class conditional probability and posterior probability
n2 = size(target, 1);
postProbPos = zeros(n2, 1); % posterior probability of being positive class 
for i=1:n2 % each target instance
    condProbPos = ones(m,1); % Conditional probability of Defective class
    condProbNeg = ones(m,1); % Conditional probability of non-defective class
    for d=1:m % each feature
        tempPos = sum(wIns((matSrc(:,end)==1)&(matSrc(:,d)==matTar(i,d)),1)); % Consider the source instances which have label 1 and their d-th feature have the same value as matTar(i,d) 
        tempNeg = sum(wIns((matSrc(:,end)==0)&(matSrc(:,d)==matTar(i,d)),1));
        
        condProbPos(d) = (1+tempPos)/(sum(wIns(matSrc(:,end)==1,1))+numel(unique(matSrc(:,d))));
        condProbNeg(d) = (1+tempNeg)/(sum(wIns(matSrc(:,end)==0,1))+numel(unique(matSrc(:,d))));
    end
    
    % Posterior probability
    postProbPos(i,1) = priorProb(2)*prod(condProbPos.^wFea) / (priorProb(1)*prod(condProbNeg.^wFea)+priorProb(2)*prod(condProbPos.^wFea)); 
    
end


try
    [MCC, F1, AUC] = Performance(target(:,end), postProbPos); % Call self-defined Performance()
catch
    F1=nan;AUC=nan;MCC=nan;
end

end


function arff = mat2ARFF(data, type)
% Summary of this function goes here: 
%   Detailed explanation goes here
% INPUTS:
%   (1) data - a n*(d+1) matrix where the last column is independent variable.
%   (2) type - a string, 'regression' or 'classification'
% OUTPUTS:
%   arff     - an ARFF file

%javaaddpath('D:\Program Files\Weka-3-8-4\weka.jar');
if ~exist('type','var')||isempty(type)
    type = 'regression';
end
label = cell(size(data,1),1);
if strcmp(type, 'classification')
    temp = data(:,end);
    for j=1:size(data,1)
        if (temp(j)==1)
            label{j} = 'true';
        else
            label{j} = 'false';
        end
    end %{0,1}--> {false, true}
else 
    label = num2cell(data(:,end));
end
featureNames = cell(size(data,2),1);
for j=1:(size(data,2)-1)
    featureNames{j} = ['X', num2str(j)];
end
featureNames{size(data,2)} = 'Defect';
arff = matlab2weka('data', featureNames, [num2cell(data(:,1:end-1)), label]);
end

function score_KL = KLDiv(vec1, vec2)
%KLDIV Summary of this function goes here: Compute Kullback-Leibler Divergence of Two variables.
%   Detailed explanation goes here
% INPUTS:
%   (1) vec1 - a coumn vector of size n where vec1(i) denotes the
%   probability of vec1=i.
%   (2) vec2 - a coumn vector of size n where vec2(j) denotes the
%   probability of vec2=j.
% OUTPUTS:
%   score_KL - a number, score_KL=0 when vec1=vec2
%


if length(vec1)~=length(vec2)
    score_KL = 1/eps; % Any large enough number
else
    % Compute Kullback-Leibler Divergence
    score_KL = sum(sum(vec1.* log(eps + vec1./(vec2+eps))));
end

end
