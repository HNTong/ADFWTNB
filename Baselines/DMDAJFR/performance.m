function [f_measure,gmean,balance,MCC] = performance(Y_pesudo, Y_ture)
% input
    %%% Y_ture£º a column vetor, each row is an instance's class label {-1,1}, -1 denotes nondefective, 1 denotes defective.
%   %%% Y_pesudo£º prediction label£¬ which has the same size as actual_label.
% output: 
    %%% four evaluation indicators f_measure,gmean,balance,MCC. 

acc=length(find(Y_pesudo==Y_ture))/length(Y_ture);
[A,~] = confusionmat(Y_ture, Y_pesudo); % confusion function
tp=A(2,2); 
fn=A(2,1);
tn=A(1,1);
fp=A(1,2);
precision = A(2,2)/(A(1,2) + A(2,2));
recall = A(2,2)/(A(2,1)+ A(2,2)); 

specificity=A(1,1)/(A(1,1)+A(1,2));
gmean=sqrt(specificity*recall);
b=2;
pd=recall;
pf=1-specificity;
f_measure=((1+b^2) * precision * recall)./(b^2*precision + recall);
MCC = (tp*tn - fp*fn)./ sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
balance=1-(sqrt((1-pd)^2+(0-pf)^2)/sqrt(2));
end

