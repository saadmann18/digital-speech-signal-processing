clc;
clear all;
close all;


%original signal parameters

F_frequency_analog = 60;
A_amplitude = 1;

ta_analog = 0 : 1/2000 : 2/F_frequency_analog ; 

% original signal

xa_analog = A_amplitude * sin ( 2 * pi * F_frequency_analog * ta_analog );       %%close to continuous approximation

figure(1);
subplot(311);
plot( ta_analog , xa_analog , 'lineWidth' , 3 );
legend('analog signal-original');

%%sampling
% sampling parameters

Fs_sampling_freq = 1000;
Ts_sampling_interval = 1/Fs_sampling_freq;

f_discrete_frequency = F_frequency_analog/ Fs_sampling_freq ; 


ts_sampling_time_sequence = 0 : Ts_sampling_interval : 2/F_frequency_analog ;

ns_sample_number = 0 : length (ts_sampling_time_sequence)-1 ; 


%sampled signal

xs_sampled_ver1 = A_amplitude * sin ( 2 * pi * f_discrete_frequency * ns_sample_number );

subplot(313);
stem ( ns_sample_number , xs_sampled_ver1 );
legend('sampled signal vs sample number');

xs_sampled_ver2 = A_amplitude * sin ( 2 * pi * F_frequency_analog * ts_sampling_time_sequence) ; 

subplot(312)
stem ( ts_sampling_time_sequence, xs_sampled_ver2 );
legend('sampled signal vs time');

%%reconstruction
recovered_signal = interp1(ts_sampling_time_sequence , xs_sampled_ver2 , ta_analog ,'spline') ; 

figure(2);
subplot(311);
plot ( ta_analog , xa_analog , 'lineWidth' , 3 );
legend('analog signal-original');

subplot(312);
stem ( ts_sampling_time_sequence, xs_sampled_ver2 );
legend('sampled signal');

subplot(313);
plot ( ta_analog , recovered_signal ) ; 
legend('reconstructed signal by interpolation');


% %%quantization
% 
% %quantization parameter
% s = xs_sampled_ver2 ; %making a copy
% 
% bit = 3 ;
%   
% %SQNR = zeros(1 , length(bit));
% 
% quant_level_number = 2 ^ bit  ; 
% quant_levels = linspace( min(xs_sampled_ver2)  , max(xs_sampled_ver2) , quant_level_number)  ; 
% 
% delta_quant_level_interval = abs(quant_levels(1) - quant_levels(2));
% 
% x_quant = zeros (1, length(xs_sampled_ver2));
% xbb = zeros (1, length(xs_sampled_ver2));
% 
% 
% %looping
% 
% for i = 1 : length(s)
%     for j = 1 : length(quant_levels)-1
%         
%         if s(i)>= quant_levels(j) && s(i)<= quant_levels(j) + delta_quant_level_interval/2
%             x_quant(i) = quant_levels(j);
%             xbb(i) = j;
%             
%         elseif s(i)<=quant_levels(j+1) && s(i)> quant_levels(j) + delta_quant_level_interval/2
%             x_quant(i) = quant_levels(j+1);
%             xbb(i) = j+1;
%         end
%             
%     end
% end
% 
% 
% xbin= dec2bin(xbb); % Encoded signal
% %disp(xbin');   %binary signal
% disp(xbin);
% 
% 
% %power calculations
% Px= sum(xs_sampled_ver2.^2)/ length(xs_sampled_ver2); % Sampled signal power
% 
% en = xs_sampled_ver2 - x_quant;
% Pn = sum(en.^2)/ length(en); % Noise power
% 
% %SQNR calculation
% SQNR = 10*log10(Px/Pn)
% fprintf('\n');
% SQNR_formula= 1.76 + 6.02*bit




