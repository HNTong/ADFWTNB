function nn = nnsetup(architecture)
%NNSETUP creates a Feedforward Backpropagate Neural Network
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers.
% INPUT:
%   architecture - a vector consisting of some integers, e.g., [input_size, 1st_hidden_size, 2nd_hidden_size, ..., output_size].
% OUTPUT:
%

    % Parameters setting
    nn.size   = architecture;
    nn.n      = numel(nn.size);
    nn.activation_function              = 'sigm';       %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 0.8;            % 2, learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = 0.5;          %  Momentum
    nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = 0;            %  L2 regularization
    nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
    nn.sparsityTarget                   = 0.05;         %  Sparsity target
    nn.inputZeroMaskedFraction          = 0;            %  Only used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
    nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'

    for i = 2 : nn.n  % 1st is input layer 
        % Initialize weights and weights' momentum
        nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1))); %nn.size(i - 1)+1: +1 for [bias, W]
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
        
        % Only be used for 'sparsity', see nnff()
        nn.p{i}     = zeros(1, nn.size(i));   
    end
end
