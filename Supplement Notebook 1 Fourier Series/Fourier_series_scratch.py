#!/usr/bin/env python
# coding: utf-8

# # Fourier Series

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets.widgets import interact, IntSlider, FloatSlider

# In[ ]:
# ## Joint Definitions for Visualizations

T = 3
L = 1.5
t = np.linspace(-L*T, L*T, 6000)

# number of fourier coefficients used
K_max = 50
k_c = np.arange(K_max)
# Cosine kernel as matrix (for fourier series)
Cos = np.cos(2*np.pi/T*(t[:, None] @ k_c[None ,:]))
Cos[:, 0] = 0.5

FONT_SIZE = 16
TFONT_SIZE = 14


# In[ ]:


# function to plot a finite fourier series from the coefficients
# and the reference signal
def plot_fourier_series(a_k, x_ref, t, T):
    # Fourier sum as a matrix product
    x_c = Cos[:, :a_k.shape[0]] @ a_k
    
    plt.axhline(c='C7')
    plt.plot(t/T, x_ref, alpha=0.5, lw=3)  ## plotting square pulses
    plt.plot(t/T, x_c, 'r', lw=2);         ##plotting Cos karnels
    plt.axvline(c='C7')
    plt.grid(True);
    plt.ylabel('$x(t)$', fontsize=FONT_SIZE)
    plt.xlabel('$t/T$', fontsize=FONT_SIZE)
    plt.xticks(fontsize=TFONT_SIZE)
    plt.yticks(fontsize=TFONT_SIZE)
    plt.ylim((-0.2, 1.2))
    plt.xlim((-1.5,1.5))


# In[ ]:


# function to plot a finite fourier series from the coefficients
# and the reference signal
def plot_fourier_spectrum(a_k, T):
    k = np.arange(-a_k.shape[0]+1, a_k.shape[0]) # for stem values to plot in x axis
    X_k = np.concatenate([a_k[::-1]/2, a_k[1::]/2]) # for plotting in y axis
    
    plt.axhline(c='C7')
    (markerline, stemlines, baseline) = plt.stem(k, X_k)
    plt.setp(baseline, visible=False)
    plt.setp(markerline, linewidth=3)
    plt.setp(markerline, markersize=10)
    plt.axvline(c='C7')
    plt.grid(True);
    plt.ylabel('$X_k$', fontsize=FONT_SIZE)
    plt.xlabel('$k$', fontsize=FONT_SIZE)
    plt.xticks(fontsize=TFONT_SIZE)
    plt.yticks(fontsize=TFONT_SIZE)
    plt.ylim((-0.5, 1.2))
    plt.xlim((-50, 50))


# In[ ]:
    
# function to repeat a single pulse
# (signal must be time dependent)
def rep_pulse(signal, t, T):
    l = np.arange(-np.floor(L), np.floor(L)+1)
    return np.sum(signal(t[:,None] - l[None, :]*T), 1)


# In[ ]:


# Function to plot Fourier sum together with repeated pulse
def plot_signal(a_k, ref_pulse, t, T):
    x_ref = rep_pulse(ref_pulse, t , T)
    
    plt.figure(figsize=(16, 12))
    plt.subplot(211)
    plot_fourier_series(a_k, x_ref, t, T)
    plt.subplot(212)
    
    plot_fourier_spectrum(a_k, T)


# In[ ]:

# rectangular pulse
def rect(t, T):
    return 1.0*(np.abs(t/T) < 1/2) if T > 0 else t*0


# In[ ]:
#calling overmentioned functions with the parameters for rectangular wave
@interact(K=IntSlider(min=1, max=K_max, step=1, value=10), 
          D=FloatSlider(min=0, max=1, step=0.05, value=0.5))
def plot_rect(K, D):
    k = np.arange(K)
    Rect_k = 2*D*np.sinc(k*D)
    # the lambda expression fixes the duration argument
    plot_signal(Rect_k, lambda t: rect(t, D*T), t, T)
    plt.ylim((-0.2, 1.1*D))

# In[ ]:
    # triangular pulse:
def tri(t,T):
    return np.maximum(1-np.abs(t/T), 0) if T > 0 else t*0

# In[ ]:
@interact(K=IntSlider(min=1, max=K_max, step=1, value=4), 
          D=FloatSlider(min=0, max=1, step=0.05, value=0.5))
def plot_triag(K,D):
    k = np.arange(K)
    Tri_k = D*np.sinc(k*D/2)**2
    # the lambda expression fixes the duration argument
    plot_signal(Tri_k, lambda t: tri(t,T*D/2), t,T)
    plt.ylim((-0.1, 0.6))    

