function [Y_tar_pseudo] = Pseudolable(src_X, tar_X, src_labels)
%% KNN classifier
    knn_model=fitcknn(src_X,src_labels,'NumNeighbors',1);
    Y_tar_pseudo=knn_model.predict(tar_X);
%% SVM classifier
    %xr=src_X'; % xr instances * feature
    %bestC = 1./mean(sum(xr.*xr,2));
    %model = svmtrain(src_labels,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
    %xe=tar_X';
    %[Y_tar_pseudo,accuracy] = svmpredict(tar_labels,xe,model);
%% Logistic Regression classifier 
  % Y=(src_labels+1)*0.5; %[-1,1] vs[0,1]
   %model=glmfit(src_X,Y,'binomial', 'link', 'logit'); % 
   %predlabel=glmval(model,tar_X, 'logit');
   %Y_pseudo=double(predlabel>=0.5);
   %Y_tar_pseudo=Y_pseudo*2-1;
   end
