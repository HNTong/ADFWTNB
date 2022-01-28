function dbn = dbnsetup(dbn, x, opts)
%DBNSETUP Summary of this function goes here: Set Up a Deep Belief Network.
%   Detailed explanation goes here
% INPUTS:
%   (1) dbn  - a struct;
%   (2) x    - a n*d matrix where d is the number of features;
%   (3) opts - a struct including numepochs, batchsize, momentum, alpha...;  
% OUTPUT:
% 

    n = size(x, 2); % number of features in input data
    dbn.sizes = [n, dbn.sizes]; % [input_size, 1st_hidden_size, ...]

    for u = 1 : numel(dbn.sizes) - 1 % Each RBM (Restricted Boltzmann Machine)
        dbn.rbm{u}.alpha    = opts.alpha; 
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u)); % Connection weights of u-th RBM
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);     % Offset for input layer of u-th RBM
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1); % Offset for hidden layer of u-th RBM
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
