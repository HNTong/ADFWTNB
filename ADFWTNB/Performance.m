function [MCC, Balance, AUC] = Performance(actual_label, probPos)
%PERFORMANCE Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) actual_label - The actual label, a column vetor, each row is an instance's class label.
%   (2) predict_label - The predicted label, a column vetor, each row is an instance label.
%   (3) probPos - The probability of being predicted as postive class.
% OUTPUTS:
%   performance measures.




assert(numel(unique(actual_label)) > 1, 'Please ensure that ''actual_label'' includes two or more different labels.'); % 
assert(length(actual_label)==length(probPos), 'Two input parameters must have the same size.');


predict_label = double(probPos>=0.5);

cf=confusionmat(actual_label,predict_label);
TP=cf(2,2);
TN=cf(1,1);
FP=cf(1,2);
FN=cf(2,1);

PD=TP/(TP+FN);
PF=FP/(FP+TN);
Precision=TP/(TP+FP);
F1=2*Precision*PD/(Precision+PD);
[X,Y,T,AUC]=perfcurve(actual_label, probPos, '1');% 
MCC = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));
Balance = 1 - sqrt(((0-PF)^2+(1-PD)^2)/2);


end

