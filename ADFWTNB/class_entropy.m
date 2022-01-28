function [s_ent, class_number] = class_entropy(classes)

% INPUT         
% 	(1) classes     - vector full of class instances
% OUTPUT        
%   (1) s_ent       - class entropy
%   (2)class_number - how many classes were there
% NOTES
%   calculate the class entropy of several points
%   remember that 'classes' deals with true classes, not discretized labels
%
% Lawrence David - 2003.  lad2002@columbia.edu


[class_list] = unique(classes);    % Get different class labels
class_number = length(class_list); % Number of different classes
s_count      = 0;

for i = 1:class_number
    s_count(i) = sum(classes==class_list(i)); % count how many different class instances you have
end

% probability of each class
s_temp = s_count/(length(classes)+realmin);   % realmin returns the smallest positive normalized floating point number (2.2251e-308) in IEEE double precision.

if s_temp == 0
    s_ent = 0;
else
    s_ent = -sum(s_temp.*log2(s_temp));
end