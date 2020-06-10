#!/usr/bin/env python
# coding: utf-8

# # From Fourier Transform to FFT

# ## Discrete-Time Fourier Transform

# In[6]:
import numpy as np

# In[7]:

def fft(x):
    M = len(x)
    if M <= 1:
        return x
    if M % 2 != 0:
            raise ValueError('{M} is not a power of 2!'.format(M=M))
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        twiddle = np.exp(-2j*np.pi/M*np.r_[0:M])
        return np.concatenate((
            X_even + twiddle[0:M//2]*X_odd,
            X_even + twiddle[M//2:M]*X_odd
        ))

# In[8]:

# ## Test Cases

# Numpy provides a function `np.isclose` which performs an element-wise floting-point "comparison" of two arrays by respecting a given precision. The result is a boolean array which can be fed into `np.all` to evaluate whether the comparison holds for all elements.   
# This call is placed in an assert statement which raises an `AssertionError` on failure.
        
def rms(x, x_ref):
    return np.linalg.norm(x - x_ref)/np.sqrt(len(x))


# In[9]:

# ### Test Case: Compare with result calculated by hand
x = np.r_[0:4]
X = fft(x)
X_hand = np.array([6, -2+2j, -2, -2-2j])

assert np.all(np.isclose(X, X_hand)), 'Failed: Not close to calculation by hand!'
print(f'error energy:  {np.sum(np.abs(X - X_hand)**2)}')

# In[11]:
# ### Test Case: Compare to numpy fft implementation

# As an alternative, a reference implementation can be used for comparison. This is of course only possible if the exact same function was implemented alread.

x = np.r_[0:16]
X = fft(x)
X_np = np.fft.fft(x)

assert np.all(np.isclose(X, X_np)), 'Failed: Not close to numpy implementation!'
print(f'error energy:  {np.sum(np.abs(X - X_np)**2)}')


# In reality, one needs to test for much more values:
# - What happens if x is not a `np.array`? 
# - What happens if x is not even an iterable?
# - What happens for the zero signal?
# - What happens for a complex signal?
