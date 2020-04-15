clc
clear all
close all
fc=1/8;
n=-40:40;
x=sinc(2*fc*n);
plot(x)
M=101;
w=linspace(-pi,pi,M);
xw=zeros(1,length(w))
for i1=1:length(w)
    for i2=1:length(n)
    xw(i1)=xw(i1)+x(i2)*exp(-j*w(i1)*n(i2));
end
end
figure(2)
    plot(w,abs(xw))
    
%reconstructed signal
xn=zeros(1,length(n))
for i1=1:length(n)
    for i2=1:length(w)
        xn(i1)=xn(i1)+(1/2*pi)*xw(i2)*exp(j*w(i2)*n(i1));
    end
end
figure(3)
plot(n,xn)
    
