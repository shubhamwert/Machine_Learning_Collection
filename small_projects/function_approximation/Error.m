function g = Error(y,pred)
temp=(y-pred).^2;
temp2=temp/size(y,1);
g=sqrt(sum(temp2));



end