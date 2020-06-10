#!/usr/bin/env python
# coding: utf-8

# # From Fourier Transform to FFT

# ## Discrete-Time Fourier Transform

# In[6]:

import numpy as np

# In[7]:

x = np.r_[0:4]
M = len(x)
X_even = x[::2]
X_odd = x[1::2]
twiddle = np.exp(-2j*np.pi/M*np.r_[0:M])
X =  np.concatenate((
            X_even + twiddle[0:M//2]*X_odd,
            X_even + twiddle[M//2:M]*X_odd
        ))

# In[8]:

def rms(x, x_ref):
    return np.linalg.norm(x - x_ref)/np.sqrt(len(x))

# In[9]:


X_hand = np.array([6, -2+2j, -2, -2-2j])

assert np.all(np.isclose(X, X_hand)), 'Failed: Not close to calculation by hand!'
print(f'error energy:  {np.sum(np.abs(X - X_hand)**2)}')

# In[11]:

x = np.r_[0:16]
M = len(x)
if M <= 1:
    print(x)
if M % 2 != 0:
        raise ValueError('{M} is not a power of 2!'.format(M=M))
else:
    X_even = x[::2]
    X_odd = x[1::2]
    twiddle = np.exp(-2j*np.pi/M*np.r_[0:M])
X =  np.concatenate((
            X_even + twiddle[0:M//2]*X_odd,
            X_even + twiddle[M//2:M]*X_odd
        ))
X_np = np.fft.fft(x)

assert np.all(np.isclose(X, X_np)), 'Failed: Not close to numpy implementation!'
print(f'error energy:  {np.sum(np.abs(X - X_np)**2)}')


