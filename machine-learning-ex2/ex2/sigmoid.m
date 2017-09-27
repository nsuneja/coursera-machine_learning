function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

nrows = size(z)(1)
ncols = size(z)(2)

for i=1:nrows
   for j=1:ncols
       g(i,j) =  1/(1 + power(e, -z(i,j)))
   end
end




% =============================================================

end
