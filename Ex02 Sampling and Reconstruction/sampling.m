clc
clear all
close all

F_analog=60;
fs=[200,500,800,1000];
k=length(fs);
for i=1:k
    ta=0:1/fs(i):.2;
    n=0:length(ta)-1;
    xs=20*sin(20*pi*1/fs(i)*n)+30*sin(30*pi*1/fs(i)*n);
    subplot(4,1,i)
    stem(n,xs)
    
end
    
    