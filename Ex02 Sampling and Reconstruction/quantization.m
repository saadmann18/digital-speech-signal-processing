%201716019
%ALI-EMEM-AL-BADI
clc;
clear all;
close all;
A=1;
F_analog=60;
Fs=1000;
bit=[3 4 5];
f=F_analog./Fs;
ta=0:1/2000:2/F_analog;
ts=0:1/Fs:2/F_analog;
n=0:length(ts)-1;
x=A*cos(2*pi*F_analog*ta);
xs=A*cos(2*pi*F_analog*ts);
xn=A*cos(2*pi*f*n);
figure(1);
subplot(311)
plot(ta,x)
title('original signal')
subplot(312)
stem(ts,xs)
subplot(313)
stem(n,xn)
title('sampled signal')

xr=interp1(ts,xn,ta,'spiline');
figure(2);
subplot(311)
plot(ta,x)
title('original signal')
subplot(312)
stem(ts,xs)
subplot(313)
plot(ta,xr)
title('reconstructed signal')

for k=1:length(bit)
R=bit(k);
L=2^R;
delta=(abs(max(xn)-min(xn)))/(L-1);
q_level=min(xn):delta:max(xn);
q_e=0:length(q_level);

for i=1:length(xn)
    for j=1:length(q_level)-1
        if xn(i)>=q_level(j) && (xn(i)<=q_level(j)+(delta/2))
            xq(i)=q_level(j)
          
        else if  xn(i)<=q_level(j+1) && (xn(i)>=q_level(j)+(delta/2))
                  xq(i)=q_level(j+1)
            
                  
            end
        end
    end
end


figure(3)
s(k)=subplot(3,1,k)
plot(n,xn,'r*')
hold on
stairs(n,xq,'LineWidth',2.5)



signalpower=10*log10(sum(xn.^2)/length(xn));
xnoise=(xn-xq);
noisepower=sum(xnoise.^2)/length(xnoise);
sqnr(k)=10*log10(signalpower/noisepower);
end
title(s(1),'Quantization for 3 bit')
title(s(2),'Quantization for 4 bit')
title(s(3),'Quantization for 5 bit')
disp(sqnr)



