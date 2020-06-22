%==========================================================================
% This function creates a Radial basis function network for the purpose of
% classification
% Inputs:
%           x: n-by-m features data. n indicates number of instances and m
%       implies the dimensions of the space
%           t: n-by-1 one-hot encoded targets vector
%           spread: Width of all basis functions
%           c: number of classes in target
%           k: number of centers
%           nnh: The number neurons in hidden layer
% Output:   

% This code has been developed for educational purposes.
%==========================================================================

function [res] = rbf(X, t, X_test, spread, k)
    
    W = rand(k, size(X,1));
    
    if size(X,2) < k
        error("Number of centers must not be less than num of classes");
    end
    
    [~, C] = kmeans(X, k);
    hold on
    scatter(C(:,4), C(:,3),50, 'filled', 'yellow')
    
    % calculate phi (radial distance)
    Phi_train = radialDist(X, C, spread);
    
    % solve the equation Phi * W = T
    W = pinv(Phi_train) * t;
    
    Phi_test = radialDist(X_test, C, spread);
    p = Phi_test * W;
    [~, res] = max(p, [],2);

end

function [D] = radialDist(X, C, spread)
    D = [];
    for i=1:size(X,1)
        D = [D; dists(X(i,:),C, spread)];
    end
    
end

function d = dists(x,C, spread)
    
    T = repmat(x, size(C,1), 1);
    
    d = exp(-vecnorm((T - C)').^2 / 2*spread^2);
    
    
end
