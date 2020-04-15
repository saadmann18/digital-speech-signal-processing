clear all
clc
x = 0:(2*pi)/1000:pi-(2*pi)/1000;
diraccomb = zeros(size(x));
diraccomb(1) = 1;
diraccomb = repmat(diraccomb,1,10);
xnew = 0:(2*pi)/1000:(5000*(2*pi)/1000)-(2*pi)/1000;