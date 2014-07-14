function x_n = mapNonLinear(x,d)
% Inputs:
% x - a single column vector (N x 1)
% d - integer (>= 0)
% Outputs:
% x_n - (N x (d+1))

x_n = ones(size(x,1), (d + 1));
for i = 1 : size(x,1)
    for j = 1 : (d + 1)
        x_n(i, j) = x(i, 1) .^ (j - 1);
    end
end