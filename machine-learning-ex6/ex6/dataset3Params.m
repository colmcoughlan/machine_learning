function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
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

err_max = 999999999.9;

C_test = 0.01;
while(C_test<30)
	sigma_test = 0.01;
	while(sigma_test<30)
		model= svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test)); 
		predictions = svmPredict(model, Xval);
		err = mean(double(predictions ~= yval));
		if(err < err_max)
			err_max = err;
			C = C_test;
			sigma = sigma_test;
		endif
		sigma_test = sigma_test * 3;
	endwhile
	C_test = C_test * 3;
endwhile





% =========================================================================

end
