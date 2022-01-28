function [X_src_globalx,X_tar_globalx,W] =GMDA(X_src,Y_src,X_tar,Y_tar_pseudo,parameter)
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
 %% 
n_src=size(X_src,1);
n_tar=size(X_tar,1);
X = [X_src',X_tar'];
[m,n] = size(X);
label=unique(Y_src);
Num_Class = length(unique(Y_src));
[num_sub_src,sub_src]=subData(X_src,Y_src);
[num_sub_tar,sub_tar]=subData(X_tar,Y_tar_pseudo);
for k=1:parameter.layers
    X=[X;ones(1,n)];
    X = X*diag(sparse(1./sqrt(sum(X.^2))));
    S=X*X'; % scatter matrix
    q=ones(m+1,1)*(1-parameter.noises);
    q(end)=1;
    Q1 = S.*(q*q');
    Q1(1:m+2:end) = q.*diag(S); %dag ¶Ô½ÇÏß
    P=S(1:end-1,:).*repmat(q',m,1); %m*(m+1)
    
    Hsc=[];
    Htc=[];
    Hstc=[];
    for i=1:Num_Class
        n_src_c=num_sub_src(i);
        Hsc1=(1/n_src_c.^2).*(ones(n_src_c,1)*ones(n_src_c,1)');
        n_tar_c=num_sub_tar(i);
        Htc1=(1/n_tar_c.^2).*(ones(n_tar_c,1)*ones(n_tar_c,1)');
        Hstc1=(-1/(n_tar_c*n_src_c)).*(ones(n_src_c,1)*ones(n_tar_c,1)');
        Hsc=blkdiag(Hsc, Hsc1);
        Htc=blkdiag(Htc, Htc1);
        Hstc=blkdiag(Hstc, Hstc1);
    end
    %%  global feature learning
    Hsm=(1/n_src.^2).*(ones(n_src,1)*ones(n_src,1)');
    Htm=(1/n_tar.^2).*(ones(n_tar,1)*ones(n_tar,1)');
    Hstm=(-1/(n_tar*n_src)).*(ones(n_src,1)*ones(n_tar,1)');
    
    M0=[Hsm,Hstm;Hstm',Htm];
    M1=[Hsc,Hstc;Hstc',Htc];
    M=M0+M1;
    M=M/norm(M,'fro');
    MMD=X*M*X';
    Q2= MMD.*(q*q');
    Q2(1:m+2:end) = q.*diag(Q2);
    reg = parameter.lambda*eye(m+1);
    reg(end,end)=0;
    W= P/(Q1+reg +parameter.beta*Q2);  %%global feature  matrix mapping
    global_all=W*X;
    global_all=tanh(global_all);
    global_all= global_all*diag(sparse(1./sqrt(sum(global_all.^2))));
    X_src_globalx=global_all(:,1:n_src)';  %% source global  feature representations
    X_tar_globalx=global_all(:,n_src+1:end)';  %%target global  feature representations
    [Y_tar_pseudo] = Pseudolable(X_src_globalx, X_tar_globalx, Y_src);% Update pseudo label
    X=global_all;
end
  
end

