#!/usr/bin/env python
# coding: utf-8

# In[4]:

import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from matplotlib.offsetbox import AnchoredText


from dssplib import plot_time_domain_signal, stft, plot_stft

plt.rcParams['figure.figsize'] = (15.0, 4.0)

# In[4]:

sample_rate, x_in = wav.read("fg_nt_upb_16k.wav")
x_in = x_in.astype(np.float)/np.abs(x_in).max()

plot_time_domain_signal(x_in, sample_rate,
                 title="Fachgebiet Nachrichtentechnik der Universität Paderborn",
                 ylabel=r'$x_\mathrm{in}(t)$')

# In[5]:

#Good setup at N_exp = 8, B_exp = 4
N_exp = 9
B_exp = 7
N = int(2**N_exp)
B = min(int((2**B_exp)), N)
X = stft(x_in, N, B)
fft_length, num_frames = X.shape
plt.imshow(np.abs(X[:fft_length//4+1]), origin='lower', aspect='auto');
plt.xlabel(r'$m$', fontsize=15)
plt.ylabel(r'$k$', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
at = AnchoredText('N={N:d}\nB={B:d}'.format(N=N, B=B), 
                  prop={'size': 15}, loc=2)
plt.gca().add_artist(at)
plt.show()

# In[6]:

floor_val = -66
N_exp =  9
B_exp = 5
N = int(2**N_exp)
B = min(int((2**B_exp)), N)
X = stft(x_in, fft_length=1024, window_length=N, frame_shift=B)
plot_stft(X, frame_shift=B, sample_rate=sample_rate, floor_val=floor_val)

at = AnchoredText('N={N:d}\nB={B:d}'.format(N=N, B=B), 
                  prop={'size': 15}, loc=2)
plt.gca().add_artist(at)
plt.show()
