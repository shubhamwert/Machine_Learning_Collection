function g = sigmoidGradient(z)


g = zeros(size(z));

m=sigmoid(z);
g=m.*(1-m);
















end
