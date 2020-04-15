clear all;
close all
clc

f = 5;
T = 1/f;

w1 = -2*pi/T:0.01:2*pi/T;
y1 = 1-abs(w1/(pi/T))

w2 = -2*pi/T:0.01:2*pi/T;
y2 = 1-abs(w2/((2*pi)/T))


plot(w1/(pi/T),T*y1, 'r')
hold on
plot(w2/(pi/T),T*y2)
grid on