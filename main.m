%% INITIALIZATION & CONTANTS
NUM_OF_CENTERS = 30;
SPREAD = 0.1;

%% DATA PREPARATION
iris=csvread('iris.csv');

y_temp=iris.variety;

% convet to one-hot encoding
y = zeros(size(y_temp,1),3);
y(:,1) = (y_temp=="Setosa");
y(:,2) = (y_temp=="Versicolor");
y(:,3) =(y_temp=="Virginica");
clear y_temp

iris.variety = [];
X = table2array(iris);

%% TRAIN & TEST

test_P = 0.3;
[m,n] = size(X) ;
idx = randperm(m);  % for shuffeling data

scatter(X(:,4), X(:,3),50, y, 'filled');

% 3-Folding
for k=1:3
    % test data extraction
    fprintf("data number %d to %d as test\n", round((k-1)*test_P*m)+1, round(k*test_P*m))
    X_test = X(idx(round((k-1)*test_P*m)+1:round(k*test_P*m)),:); 
    y_test = y(idx(round((k-1)*test_P*m)+1:round(k*test_P*m)),:); 
    
    % train data extraction
    X_train = X; y_train = y;
    X_train(idx(round((k-1)*test_P*m)+1:round(k*test_P*m)),:) = [];
    y_train(idx(round((k-1)*test_P*m)+1:round(k*test_P*m)),:) = [];
    
    % RBF neural network
    t = rbf(X_train, y_train, X_test, 0.5, 4);
    
    % calculate accuracy
    matches = (t == max(y_test, [],2));
    
    fprintf("Accuracy: %d\n", sum(matches)/size(y,1));
end