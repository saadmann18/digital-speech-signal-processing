#!/usr/bin/env python
# coding: utf-8


# In[1]:

import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from IPython.display import Audio, display
from numpy.fft import fft, ifft

# In[2]:

def plot_signal(x, sample_rate, *args, **kwargs):
    t = np.r_[:len(x)]*1/sample_rate
    plt.plot(t, x)
    plt.axhline(c='k', alpha=0.75)
    plt.axvline(c='k', alpha=0.75)
    plt.xlabel(kwargs.get('x_label', r'$t / \mathrm{s}$'), fontsize=14)
    plt.ylabel(kwargs.get('y_label', r'$x(t)$'), fontsize=14)
    plt.grid(True)


# In[3]:

def show_signal(x, sample_rate, *args, **kwargs):
    display(Audio(x, rate=sample_rate))
    plot_signal(x, sample_rate, *args, **kwargs)


# In[4]:


def normalize(x):
    return x / np.abs(x).max()


# In[5]:


sample_rate, in_signal = read('fg_nt_upb_16k.wav')
in_signal = normalize(in_signal.astype(np.float))
show_signal(in_signal, sample_rate)


# In[6]:


sample_rate_rir, rir = read('raumimpulsantwort.wav') #room impulse response r.i.r
assert np.isclose(sample_rate_rir, sample_rate),     "The sample rates do not match!"
rir = normalize(rir.astype(np.float))
show_signal(rir, sample_rate, y_label=r'$h(t)$')

# In[9]:


out_conv = normalize(np.convolve(rir, in_signal))
show_signal(out_conv, sample_rate, y_label=r'$y_\mathrm{conv}(t)$')


# In[11]:
# TODO compare it with overlap save method copy N = L+M-1, N=DFT length, M=impulse length, L = N+1-M = block
in_signal = np.array([1,2,-1,2,3,-2,-3,-1,1,1,2,-1])
rir_M = np.array([1,2,3,0])
fft_length_N=4
shift_L = fft_length_N - len(rir) + 1    # block length B in formulas ##L
num_blocks = len(in_signal) // shift_L

rir_ft = fft(rir_M, fft_length_N)
input_block = np.zeros(fft_length_N)
out_signal = np.zeros(num_blocks * shift_L)

for m in range(num_blocks):
    input_block[:fft_length_N-shift_L] = input_block[shift_L:]  
    input_block[fft_length_N-shift_L:] = in_signal[m*shift_L:(m+1)*shift_L]
    in_ft = fft(input_block, fft_length_N)
    tmp = ifft(in_ft * rir_ft, fft_length_N).real
    out_signal[m*shift_L:(m+1)*shift_L] = tmp[fft_length_N-shift_L:]
out_signal


# In[12]:


##get_ipython().run_line_magic('timeit', 'normalize(filter_overlap_save(in_signal, rir))')


# In[14]:


out_overlap_save = normalize(out_signal)
show_signal(out_overlap_save, sample_rate, y_label=r'$y_{\mathrm{OA}}(t)$')


# In[15]:


error_signal = out_overlap_save - out_conv[:len(out_overlap_save)]
plot_signal(error_signal, sample_rate, y_label=r'$e(t)$')
print('Power of error signal: {:0.3e}'.format(np.abs(error_signal**2).mean()))


# ## Task 5.3


# ## Load a Long Room Impulse Response

# In[19]:


sample_rate_rir_long, rir_long = read('IR_cathedral_16k.wav')
assert np.isclose(sample_rate, sample_rate_rir_long),     "The sample rates do not match!"
rir_long = normalize(rir_long.astype(np.float))
show_signal(rir_long, sample_rate, y_label=r'$h_\mathrm{cathedral}(t)$')


# ## Task 5.6:
# 
# Complete the function `partitioned_block_overlap_save`:
# 1. Extend the previous algorithm to accumulate over all partitions.
# 2. Perform the convolution also in the time domain to compare the results and processing times.

# In[20]:

## def partitioned_block_overlap_save(in_signal, impulse_response, fft_length=512):

    
in_signal = np.array([1,2,-1,2,3,-2,-3,-1,1,1,2,-1])
rir = np.array([1,2,3,0])
fft_length=4
    
shift = fft_length // 2
num_blocks = len(in_signal) // shift
partitions = len(rir) // shift + 1 #, both logic works fine here
ir_length = partitions * shift

# pad with zeros, thus length is multiple of partitions
rir = np.r_[
    rir, np.zeros(ir_length - len(rir))]
# break into partitions and do fft on each
rir_ft = fft(rir.reshape((partitions, -1)), fft_length)

assert rir_ft.shape == (partitions, fft_length),  "Malformed impulse response array after fourier transformation!"

out_signal = np.zeros(num_blocks*shift)
input_block = np.zeros(fft_length)
fft_buffer = np.zeros((partitions, fft_length), dtype=np.complex128) #why complex?? 

for m in range(num_blocks):
    input_block[:-shift] = input_block[shift:]  # 8 REPLACE # TODO input_block[:fft_length-shift] = input_block[shift:] in first problem
    input_block[fft_length-shift:] = in_signal[m*shift:(m+1)*shift]

    fft_buffer[1:, :] = fft_buffer[:-1, :]
    fft_buffer[0, :] = fft(input_block, fft_length)

    out_buffer = ifft((fft_buffer * rir_ft).sum(axis=0), fft_length).real
    out_signal[m*shift:(m+1)*shift] = out_buffer[-shift:]

out_signal




# In[21]:
# #### Timing of Convolution

##get_ipython().run_line_magic('timeit', 'normalize(np.convolve(rir_long, in_signal))')


# In[25]:


out_conv_long = normalize(np.convolve(rir_long, in_signal))
show_signal(out_conv_long, sample_rate, y_label=r'$y_\mathrm{conv,long}(t)$')


# In[23]:


##get_ipython().run_line_magic('timeit', 'normalize(partitioned_block_overlap_save(in_signal, rir_long))')


# In[26]:


out_pbos = normalize(partitioned_block_overlap_save(in_signal, rir_long))
show_signal(out_pbos, sample_rate, y_label=r'$y_\mathrm{pbos,long}(t)$')


# In[27]:


error_long = out_pbos - out_conv_long[:len(out_pbos)]
plot_signal(error_long, sample_rate, y_label=r'$e_\mathrm{long}(t)$')
print("Power of long error signal: {}".format(np.abs(error_long**2).mean()))

