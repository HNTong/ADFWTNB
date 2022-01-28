function [num_sub,sub_X]=subData(Xc,Y)
% 
num_sub=[];
label=unique(Y);
  for i=1:length(label)
      index=find(Y==label(i));
      num_sub(i)=length(index);
      sub_X{i}=Xc(index,:);
  end
end

 