function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 100;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
_C = _sigma = [0.01 0.03 0.1 0.3 1 3 10 30];
e = zeros(numel(_C), numel(_sigma));
for n = 1:numel(_C);
	for m = 1:numel(_sigma);
		fprintf('Train and evaluate (on cross validation set) for\n[C, sigma] = [%f %f]\n', _C(n), _sigma(m));
		model = svmTrain(X, y, _C(n), @(x1, x2) gaussianKernel(x1, x2, _sigma(m)));
		e(n, m) = mean(double(svmPredict(model, Xval) ~= yval));
	end;
end;
[Cmin, CminI] = min(e);
[sigmamin, sigmaminI] = min(Cmin);

sigmaI = sigmaminI;
CI = CminI(sigmaI);

C = _C(CI);
sigma = _sigma(sigmaI);
% =========================================================================

end
