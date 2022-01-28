function [MCC, F1, AUC, nullIdxSrc, nullIdxTar] = DBN_CP(sourcePath, targetPath, i, srcCodeDir, tarCodeDir, saveDir)
%DBN_CP Summary of this function goes here: Implement Deep Belief Network
% for Cross-Project Defect Prediction
%   Detailed explanation goes here
% INPUTS:
%   (1) sourcePath  - a string denoteing the absolute path of source data file (.csv), e.g., 'E:\\Document\\GitHub\\SDP-Datasets\\MORPH\\CSV\\ant-1.3.csv'.
%   (2) targetPath  - a string like 'sourcePath'.
%   (3) srcCodeDir  - a string, the direction of source code of source project. 
%   (4) tarCodeDir  - a string, the direction of source code of target project.
%   (5) saveDir     - a string, the direction of file save. 
% OUTPUTS:
%   PD, PF, ...
%
% Reference:S. Wang, T. Liu, J. Nam and L. Tan, "Deep Semantic Feature
% Learning for Software Defect Prediction," in IEEE Transactions on
% Software Engineering, vol. 46, no. 12, pp. 1267-1293, 1 Dec. 2020, doi:
% 10.1109/TSE.2018.2877612.
%
% 

%% Default value
if ~exist('savePath','var')||isempty(saveDir)
    saveDir = [pwd, '\'];
end
if ~exist('srcCodePath','var')||isempty(srcCodeDir)
    srcCodeDir = 'E:/Document/Other/ICNN-PanCong/source file/';
end
if ~exist('tarCodePath','var')||isempty(tarCodeDir)
    tarCodeDir = 'E:/Document/Other/ICNN-PanCong/source file/';
end
if ~exist('i','var')||isempty(i)
    i = '';
end

%% Encoding AST nodes
py.MapToken.mapping_token(sourcePath, saveDir, srcCodeDir, i); % Generate a XX.mat file (based on source project code) under saveDir
py.MapToken.mapping_token(targetPath, saveDir, tarCodeDir, i); % Generate a YY.mat file (based on target project code) under saveDir

% load XX.mat
strs = split(sourcePath, '\\'); % Return a cell vector
if ~isempty(i)
    i = ['_', i];
end
str = strs{end};
pathSrc = [saveDir, strrep(str, '.csv',[i, '.mat'])];
srcSemX = load(pathSrc);
srcSemX = srcSemX.token;

% load YY.mat 
strs = split(targetPath, '\\'); % Return a cell vector
str = strs{end};
pathTar = [saveDir, strrep(str, '.csv',[i, '.mat'])];
tarSemX = load(pathTar);
tarSemX = tarSemX.token; % Obtain a cell where each element is a row vector

%% Padding and Labeling
[srcSemX, nullIdxSrc]= Padding(srcSemX); % Call the self-defined function 'Padding'
[tarSemX, nullIdxTar] = Padding(tarSemX);

% % Delete the constant feature 
% srcSemX = DelConsFeature(srcSemX);
% tarSemX = DelConsFeature(tarSemX);

if size(srcSemX,2) > size(tarSemX,2)
    tarSemX = Padding(tarSemX, size(srcSemX,2));
else
    srcSemX = Padding(srcSemX, size(tarSemX,2));
end

% Load CSV file
srcMetric = csvread(sourcePath, 1, 3); % read csv file from row 1 and column 3. NOTE: the initial index is 0.
srcMetric = [srcMetric(:,1:end-1), double(srcMetric(:,end)>=1)];
tarMetric = csvread(targetPath, 1, 3);
tarMetric = [tarMetric(:,1:end-1), double(tarMetric(:,end)>=1)];

% Add label column
srcSem = [srcSemX, srcMetric(:, end)];
tarSem = [tarSemX, tarMetric(:, end)];

% Remove non-existing modules according to code
srcSem(nullIdxSrc,:) = [];
tarSem(nullIdxTar,:) = [];
srcMetric(nullIdxSrc,:) = [];
tarMetric(nullIdxTar,:) = [];

%% Handling noise
srcSem = removeNoise(srcSem);
srcSemX = srcSem(:,1:end-1);
tarSemX = tarSem(:,1:end-1);

%% Train DBN
for i=1:size(srcSemX, 2)
    if length(unique(srcSemX(:,i)))==1
        srcSemX(:,i) = 0;
    end
end
for i=1:size(tarSemX, 2)
    if length(unique(tarSemX(:,i)))==1
        tarSemX(:,i) = 0;
    end
end

% Max-min normalizaton
[temp, ps] = mapminmax(srcSemX', 0, 1); 
srcSemX = temp';
% temp = mapminmax('apply', tarSemX', ps);
[temp, ps] = mapminmax(tarSemX', 0, 1); 
tarSemX = temp';

dbn.sizes = ones(1, 10) * 100; % 10 hidden layers with same size 100
opts.numepochs = 200;
opts.batchsize = 50;
opts.momentum  = 0.5;
opts.alpha     = 0.8;
dbn = dbnsetup(dbn, srcSemX, opts);
dbn = dbntrain(dbn, srcSemX, opts);

%% Obtain deep representations
nn = dbnunfoldtonn(dbn); % unfold dbn to nn
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.8;  
% nn.momentum                         = opts.momentum;
% nn.inputZeroMaskedFraction          = 0;
% nn.dropoutFraction                  = 0;
% nn.output = 'sigm';
nn = nnff(nn, srcSemX, zeros(size(srcSemX,1),1));
srcDeepSem = nn.a{end};

nn = nnff(nn, tarSemX, zeros(size(tarSemX,1),1));
tarDeepSem = nn.a{end};

%% Prediction
LR = glmfit(srcDeepSem, srcMetric(:,end), 'binomial', 'link', 'logit'); 
probPos = glmval(LR, tarDeepSem, 'logit');  % Return the probability of being positive class.
try
    [MCC， F1, AUC] = Performance(tarMetric(:,end), probPos); % Call self-defined Performance()
catch
    F1=nan;AUC=nan;MCC=nan;
end


end

function [dataPadded, nullIdx] = Padding(data, maxEleNum)
%PADDING Summary of this function goes here:
%   Detailed explanation goes here
% INPUTS:
%   (1) data - a cell vector, each element is a row vector with different number of features;
%   (2) maxEleNum - Number of features for the longest instance;
% OUTPUTS:
%   dataPadded - a matrix where each column denotes a feature.

if ~exist('maxEleNum','var')||isempty(maxEleNum)
    maxEleNum = 0; % Denote the number of elements in the longest sample
end
if iscell(data)
    if maxEleNum == 0
        for i = 1:numel(data) % each sample
            if ~isempty(data{i}) && size(data{i}, 2) > maxEleNum
                maxEleNum = size(data{i}, 2);
            end
        end
    end
    
    nullIdx = [];
    temp = zeros(numel(data), maxEleNum);
    for i = 1:numel(data)
        if ~isempty(data{i})
            ins = double(data{i});
            temp(i, 1:length(ins)) = ins;
        else
            nullIdx = [nullIdx, i];
        end
    end
    
    dataPadded = temp;
else % data is an 2D array
    temp = zeros(size(data,1), maxEleNum);
    for i = 1:size(data, 1)
        ins = double(data(i,:));
        temp(i, 1:length(ins)) = ins;
    end
    dataPadded = temp;
end
end

function cleanData = DelConsFeature(data)
%DELCONSFEATURE Summary of this function goes here: Delete the feature which always has the same one value.
%   Detailed explanation goes here
% INPUTS:
%   data - a n*d array

idxDel = [];
for i = 1:size(data, 1)
    if length(unique(data(:,i))) == 1
        idxDel = [idxDel, i]; 
    end
end
data(:,idxDel) = [];
cleanData = data;
end

