function [PD, PF, Precision, F1, AUC, Accuracy, G_measure, MCC, Balance] = ManualDown(target, idxLOC, thr)
% ManualDown - Summary of this function goes here: For a given target release, ManualDown considers a larger module as more
% defect-prone. Specifically, first rank the modules of target data descendly according to module size. Then, we classify the
% top x% modules in the ranking list as defective and the remaining modules as not defective. 
%   Detailed explanation goes here:
%   INPUTS:
%       (1) target - a n*(d+1) matrix where n and d separately denote the number of
%       samples and the number of features (i.e., metrics), the last column is the actual label;
%       (2) idxLOC - the index of LOC metric in Target;
%       (3) thr (default value 0.5) - a classifcation threshold value belonging to (0,1);
%   OUTPUTS:
%       Nine perforamnce measures including PD,PF,Precision,F1,AUC,Accuracy,G_measure,MCC, Balance.
% 
% Reference: [1] Y. Zhou, Y. Yang, H. Lu, L. Chen, Y. Li, Y. Zhao, J. Qian, and
% B. Xu,"How far we have progressed in the journey? an examination of
% cross-project defect prediction", ACM Transactions on Software
% Engineering and Methodology (TOSEM), vol. 27, no. 1, pp. 1-51, 2018.
%

% Default value
if ~exist('thr', 'var')||isempty(thr)
    thr = 0.5; % Used by Zhou et al.
end

n = size(target, 1); % The number of target instances
[~,idx] = sort(target(:,idxLOC),'descend'); % Sort target instances in descending order of LOC
marker = floor(n * thr);
predLabel = zeros(n,1); % Initialization
for i=1:n % each sample
    if i <= marker
        predLabel(i) = 1; % defective
    else
        predLabel(i) = 0; % non-defective
    end
end

[~,idxIdx] = sort(idx,'ascend');
predLabel = predLabel(idxIdx,:);

probPos = predLabel; % Label probability is needed to calculate AUC
try
    [ PD,PF,Precision,F1,AUC,Accuracy,G_measure,MCC, Balance] = Performance( target(:,end),probPos); % Invoke a self-defined function Performance().
catch
    PD=nan; PF=nan; Precision=nan; F1=nan; AUC=nan; Accuracy=nan; G_measure=nan; MCC=nan; Balance=nan;
end
end

