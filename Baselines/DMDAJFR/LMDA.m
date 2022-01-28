function [X_src_localx,X_tar_localx,Wc] =LMDA(X_src,Y_src,X_tar,Y_tar_pseudo,parameter)
 %Inputs:
    %%% X_src  :source feature matrix, n_src * m
    %%% Y_src  :source label vector, n_src * 1
    %%% X_tar  :target feature matrix, n_tar * m
    %%% Y_tar_pseudo: target pseudo-label vector, n_tar * 1
    %%%parameter:option struct
%outputs:
    %%% X_src_globalx,: source global feature representations;
    %%% X_tar_globalx:  target global feature representations,
    %%% W:  global  feature  mapping  matrix;
Xc_src=X_src;
Xc_tar=X_tar;

label=unique(Y_src);
Num_Class=length(label);

label2=unique(Y_tar_pseudo); %ADD
for j=1:parameter.layers
    X_src_localx=[];
    X_tar_localx=[];
    
    %Yc_src=[];
    for i=1:Num_Class
        index=find(Y_src==label(i));
        index2=find(Y_tar_pseudo==label2(i));%ADD
        % Yc_src=[Yc_src;Y_src(index)];
        [num_sub_src,sub_src]=subData(Xc_src,Y_src);
        [num_sub_tar,sub_tar]=subData(Xc_tar,Y_tar_pseudo);
        n_src_c=num_sub_src(i);
        Hsc1=(1/n_src_c.^2).*(ones(n_src_c,1)*ones(n_src_c,1)');
        n_tar_c=num_sub_tar(i);
        Htc1=(1/n_tar_c.^2).*(ones(n_tar_c,1)*ones(n_tar_c,1)');
        Hstc1=(-1/(n_tar_c*n_src_c)).*(ones(n_src_c,1)*ones(n_tar_c,1)');
        Mc=[Hsc1,Hstc1;Hstc1',Htc1];
        Mc=Mc/norm(Mc,'fro');
        
        X_c_src=sub_src{i};
        X_c_tar=sub_tar{i};
        Xc= [X_c_src', X_c_tar'];
        Xc=Xc*diag(sparse(1./sqrt(sum(Xc.^2))));
        [m,nc] = size(Xc);
        Xc=[Xc;ones(1,nc)];
        Sc=Xc*Xc';
        q=ones(m+1,1)*(1-parameter.noises); %ADD
        q(end)=1; %ADD
        Qc1 = Sc.*(q*q');
        Qc1(1:m+2:end) = q.*diag(Sc); %dag ¶Ô½ÇÏß
        Pc=Sc(1:end-1,:).*repmat(q',m,1); %m*(m+1)
        MMDc=Xc*Mc*Xc';
        Qc2= MMDc.*(q*q');
        Qc2(1:m+2:end) = q.*diag(Qc2);
        reg = parameter.lambda*eye(m+1);
        reg(end,end)=0;
        Wc{i}= Pc/(Qc1+reg +parameter.beta*Qc2); %%local feature  matrix mapping
        local_all=Wc{i}*Xc;
        local_all=tanh(local_all);
        local_all= local_all*diag(sparse(1./sqrt(sum(local_all.^2))));
        X_src_localx(index,1:m)=local_all(:,1:n_src_c)';    %% source local feature representations    X_src_localx(index,:)=local_all(:,1:n_src_c)';
        X_tar_localx(index2,1:m)=local_all(:,n_src_c+1:end)';  %% target local feature representations   X_tar_localx(index,:)=local_all(:,n_src_c+1:end)';
    end
    Xc_src=X_src_localx;
    Xc_tar=X_tar_localx; % Xc_tar=X_src_localx; 
    [Y_tar_pseudo]= Pseudolable(X_src_localx,  X_tar_localx, Y_src); %% % Update pseudo label
end
end

