clear all;
close all;
clc

tmax = 5;
t = 0:0.1:tmax;
f = 1;

fs = 0.5;
tsamp = 0:1/fs:tmax;

xcont = cos(2*pi*f*t);
xsamp = cos(2*pi*f*tsamp);

figure;
plot(t,xcont)
hold on 
plot(tsamp, xsamp, 'r')