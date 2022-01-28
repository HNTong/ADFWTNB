function [MCC, F1, AUC] = ADFWTNB(source, target, idxLOC)
%ADFWTNB Detailed explanation goes here
% INPUTS:
%   (1) source - a n1*(d+1) matrix, the last column is the label where 0/1
%   denotes the non-defective/defective module
%   (2) target - a n2*(d+1) matrix, the last column is the label where 0/1
%   denotes the non-defective/defective module 
%   (3) idxLOC - the index of LOC metric
% OUTPUTS:
%   Performance measures

if ~all(unique(source(:,end))==[0,1]') || ~all(unique(target(:,end))==[0,1]')
    error('The label of each sample must be 0 or 1');
end

defRatio = sum(source(:,end)==1)/size(source, 1); % Calculate the defective ratio of source dataset

sourceCopy = source;
if defRatio > 0.45 || size(source,1)<80
    rand('seed',0);
    
    [MCC, F1, AUC] = ManualDown(target, idxLOC);
    
    return
end

[MCC, F1, AUC] = DFWTNB(source, target);
end

