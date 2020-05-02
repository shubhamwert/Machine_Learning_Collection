function p=pred(X,Theta1,Theta2)
m = size(X, 1);

a1=[ones(m,1),X];
z2=(a1*Theta1');
a2=[ones(size(z2),1),sigmoid(z2)];
h_theta = sigmoid(a2*Theta2');
a3=h_theta;
p=a3;