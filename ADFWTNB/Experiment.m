

%% RQ1 (one-to-one)

clear;

dataPath = '..\Datasets\PROMISE\';

% Experiment Settings 
runtimes = 30; % the number of runnings of prediction model
percent_tt = 0.9;% the percentage of training data in source data

javaaddpath('.\weka.jar');
import weka.filters.*; 
import weka.*;

dataNames = {{'jedit-4.1','ivy-2.0','tomcat'},{'ant-1.3','ant-1.3','ant-1.3'}};

for d = 1:numel(dataNames{1})
    disp(['Data: ',num2str(d),' / ',num2str(numel(dataNames{1}))]);
       
    % Load source data    
    file1 = java.io.File([dataPath,dataNames{1}{d},'.arff']);  
    loader = weka.core.converters.ArffLoader;  
    loader.setFile(file1); 
    insts = loader.getDataSet; 
    insts.setClassIndex(insts.numAttributes()-1); 
    [sources,featureNames,targetNDX,stringVals,relationName1] = weka2matlab(insts,[]); % NOTE: For 'weka2matlab', the first label value is transformed into 0, i.e., {false,true} or {N, Y}-->{0,1}, {true,false} or {Y, N}-->{0,1}
    if strcmp(featureNames{end}, 'Y')||strcmpi(featureNames{end}, 'Yes')||strcmpi(featureNames{end}, 'true')||strcmpi(featureNames{end}, 'T')
        sources(sources(:,end)==0, end) = -1;
        sources(sources(:,end)==1, end) = 0;
        sources(sources(:,end)==-1, end) = 1;
    end
	sources = [sources(:, 1:end-1), double(sources(:, end)>0)]; 
    
    % Load target data 
    file2 = java.io.File([dataPath,dataNames{2}{d},'.arff']);  
    loader = weka.core.converters.ArffLoader;  
    loader.setFile(file2);  
    insts = loader.getDataSet; 
    insts.setClassIndex(insts.numAttributes()-1);
    [targets,featureNames,targetNDX,stringVals,relationName2] = weka2matlab(insts,[]); 
	if strcmp(featureNames{end}, 'Y')||strcmpi(featureNames{end}, 'Yes')||strcmpi(featureNames{end}, 'true')||strcmpi(featureNames{end}, 'T')
        targets(targets(:,end)==0, end) = -1;
        targets(targets(:,end)==1, end) = 0;
        targets(targets(:,end)==-1, end) = 1;
    end
    targets = [targets(:, 1:end-1), double(targets(:, end)>0)];
    
    
    % Remove duplicated instances
    sources = unique(sources,'rows','stable');
    targets = unique(targets,'rows','stable');
    
    % Remove instances having missing values
    [idx_r idx_c] = find(isnan(sources));
    sources(unique(idx_r),:) = [];
    [idx_r idx_c] = find(isnan(targets));
    targets(unique(idx_r),:) = [];
    
    % LOC of target data
    idxLOC = 0;
    if strcmp(dataNames{2}{d} , 'EQ')||strcmp(dataNames{2}{d} , 'JDT')||strcmp(dataNames{2}{d} , 'Lucene')||strcmp(dataNames{2}{d} , 'Mylyn')||strcmp(dataNames{2}{d} , 'PDE')
        idxLOC = 26;
    else
        idxLOC = 11; % For ReLink, PROMISE,
    end

    % Shuffle
    if runtimes >= 1
        rand('state',0);
        sources = sources(randperm(size(sources,1),size(sources,1)),:); 
        rand('state',0); 
        targets = targets(randperm(size(targets,1),size(targets,1)),:);
    end
    
    
    % Predefine
    AUC_fmt=zeros(runtimes,1);MCC_fmt=zeros(runtimes,1);F1_fmt=zeros(runtimes,1);
    AUC_md=zeros(runtimes,1);MCC_md=zeros(runtimes,1);F1_md=zeros(runtimes,1);
    AUC_adfw=zeros(runtimes,1);MCC_adfw=zeros(runtimes,1);F1_adfw=zeros(runtimes,1);
    AUC_dmda=zeros(runtimes,1);MCC_dmda=zeros(runtimes,1);F1_dmda=zeros(runtimes,1);
    
    % targetsCopy = targets;
    sourcesCopy = sources;
    
	% Just for DMDAJFR
    paramaters.layers = 6; paramaters.noises = 0.5; paramaters.lambda = 0.01; 
    if strcmp(dataNames{1}{d}, 'ant-1.3')
        paramaters.layers = 4; paramaters.noises = 0.5; paramaters.lambda = 0.01;   
    end
    if strcmp(dataNames{1}{d}, 'camel-1.6')
        paramaters.layers = 6; paramaters.noises = 0.6; paramaters.lambda = 1;   
    end
    if strcmp(dataNames{1}{d}, 'ivy-2.0')
        paramaters.layers = 6; paramaters.noises = 0.6; paramaters.lambda = 1;   
    end
    if strcmp(dataNames{1}{d}, 'jedit-4.1')
        paramaters.layers = 4; paramaters.noises = 0.5; paramaters.lambda = 0.01;   
    end
    if strcmp(dataNames{1}{d}, 'poi-2.0')
        paramaters.layers = 4; paramaters.noises = 0.5; paramaters.lambda = 0.01;   
    end
    if strcmp(dataNames{1}{d}, 'xalan-2.4')
        paramaters.layers = 4; paramaters.noises = 0.5; paramaters.lambda = 0.01;   
    end
    if strcmp(dataNames{1}{d}, 'xerces-1.2')
        paramaters.layers = 6; paramaters.noises = 0.6; paramaters.lambda = 0.01;   
    end
    if strcmp(dataNames{1}{d}, 'synapse-1.2')
        paramaters.layers = 8; paramaters.noises = 0.7; paramaters.lambda = 0.01;   
    end
    
    for i=1:runtimes % parfor
        disp(['runtimes:',num2str(i),' / ',num2str(runtimes)]);
        
        % Divide the targets into two parts: traintarget and target (i.e., equals to target minus trainTarget)
        rand('seed',i);
        idx = randperm(size(sourcesCopy,1),round(percent_tt*size(sourcesCopy,1)));  %
        trainData = sourcesCopy(idx,:);    % Some models such as, TrAdaBoost and HYDRA need traintarget. Other models such as Burak's,Peters's filters and TNB do not use traintarget data.
        while numel(unique(trainData(:,end)))==1 % Avoid the label of traintarget has only one kind.
            idx = randperm(size(sourcesCopy,1),round(percent_tt*size(sourcesCopy,1)));
            trainData = sourcesCopy(idx,:);
        end
        sources = trainData;
        
        
        %% Proposed ADFWTNB
        disp('ADFWTNB ...');
        rand('seed',0);
        source = sources;
        target = targets;
        [MCC_adfw(i,:), F1_adfw(i,1),AUC_adfw(i,:)] = ADFWTNB(source, target, idxLOC);
        
        
        %% ManualDown
        disp('ManualDown');
        rand('seed',0);
        source = sources;
        target = targets;
        [MCC_md(i,:),F1_md(i,1),AUC_md(i,:)] = ManualDown(target, idxLOC);
        

        
        %% 8. FMT
        disp('FMT...');
        rand('seed',0);
        source = sources;
        target = targets;
        [MCC_fmt(i,:), F1_fmt(i,1), AUC_fmt(i,:)] = FMT(source, target);
        
        %% DMDAJFR
        disp('DMDAJFR...')
        rand('seed',0);
        source = sources;
        target = targets; 
        [MCC_dmda(i,:), F1_dmda(i,1), AUC_dmda(i,:)] = DMDAJFR(source, target, paramaters);
    end % End of runs
end % End of datsets
a = 1;
