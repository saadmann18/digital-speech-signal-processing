#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

from ipywidgets import interact
from scipy import signal


# # Task 1
# 
# Compare the spectra of a rectangle, Hann, Hamming and Blackman window.
# (Hint:Think about a generalized cosine window first!)   
# What is the spectrum for a triangular window?

# In[ ]:


N = 32
theta = np.r_[-np.pi:np.pi:2*np.pi/(N*8)]
n = np.r_[-5:N+5]


# In[ ]:


def make_window_plots(w, W, name='Window'):
    label_size, title_size  = 20, 25
    f, (ax_td, ax_fd) = plt.subplots(1, 2, figsize=(20, 5))
    for ax in (ax_td, ax_fd):
        ax.tick_params(axis='both', which='major', labelsize=label_size)

    # Plot the discrete time domain representation
    # of the window function
    ax_td.plot(n, w, 'o')
    ax_td.vlines(n, 0, w, lw=2, colors='C0')
    ax_td.set_xlim([n[0], n[-1]])
    ax_td.set_ylim([-0.1, 1.1])
    ax_td.set_xlabel(r'$n$', fontsize=label_size)
    ax_td.set_ylabel(r'$w(n)$', fontsize=label_size)
    ax_td.set_title('{} in time domain'.format(name),
                    fontsize=title_size)
    ax_td.axhline(c='k', lw=1.5, alpha=0.75)
    ax_td.grid(True)
    
    # Plot the spectral representation (in theta domain)
    ax_fd.plot(theta/(2*np.pi), 20*np.log10(np.abs(W)/np.max(np.abs(W))), lw=2)
    ax_fd.set_xlim([-0.5, 0.5])
    ax_fd.set_ylim([-80, 5])
    ax_fd.set_xlabel(r'$\theta/2\pi$', fontsize=label_size)
    fd_y_label = (r'$20\log\left(W\left(\mathrm{e}^{j\theta}'
                  r'\right)/N\right)/ \mathrm{dB}$')
    ax_fd.set_ylabel(fd_y_label, fontsize=label_size)
    ax_fd.set_title('{} in frequency domain'.format(name),
                    fontsize=title_size)
    ax_fd.axvline(c='k', lw=1.5, alpha=0.75)
    ax_fd.axhline(c='k', lw=1.5, alpha=0.75)
    ax_fd.grid(True)
    
    plt.show()


# In[ ]:


def generalized_cosine_window(n, N, c=[0.0, 0.0, 0.0]):
    w = # TODO
    return w


# In[ ]:


def dirichlet_spectrum(theta, N):
    # TODO
    return spectrum


# Defined as in the lecture. For more details, see [Wikipedia: Dirichlet kernel](https://en.wikipedia.org/wiki/Dirichlet_kernel)

# In[ ]:


def generalized_cosine_window_spectrum(theta, N, c=[0.0, 0.0, 0.0]):
    W = # TODO
    return W


# ## Rectangular window (often: Boxcar window)

# In[ ]:


c_rect = # TODO
w_rect = # TODO
W_rect = # TODO


# In[ ]:


make_window_plots(w_rect, W_rect, name='Rectangular window')


# ## Hann window

# In[ ]:


c_hann = # TODO
w_hann = # TODO
W_hann = # TODO


# In[ ]:


make_window_plots(w_hann, W_hann, 'Hann')


# #### Remark:
# This window is often referred to as *Hanning window*. This misconception (due to the fact that the window function is named after austrian meteorologist *Julius von Hann*) is rooted in the similarly named *Hamming window* (see below) and the fact that the inventors used the verb *hanning (a signal)* as shorthand for *applying the hann window function (to a signal)*.

# ## Hamming window

# In[ ]:


c_hamm = # TODO
w_hamm = # TODO
W_hamm = # TODO


# In[ ]:


make_window_plots(w_hamm, W_hamm, name='Hamming window')


# ## Blackman window

# In[ ]:


alpha = 0.16
c_bm = # TODO
w_bm = # TODO
W_bm = # TODO


# In[ ]:


make_window_plots(w_bm, W_bm, name='Blackman window')


# ## Triangluar (or Bartlett) Window

# In[ ]:


def bartlett_window(n, N):
    return # TODO


# In[ ]:


def bartlett_window_spectrum(theta, N):
    return # TODO


# In[ ]:


w_bart = # TODO
W_bart = # TODO


# In[ ]:


make_window_plots(w_bart, W_bart, name='Bartlett window')


# # Task 2
# 
# Examine the leakage effect with Python. To this extend, approximate the continuous-time domain signals
# 
# \begin{align*}
#     x_i(t) &= \mbox{rect} \left(\frac{t - T_M/2}{T_M}\right)\cdot\cos( 2\pi f_1 t),           &i \in \{1, 2\} \\
#           &\approx x_i(m \cdot \Delta t)
# \end{align*}
# 
# with a representation rate of $f_{rep} = 1/\Delta t = 16\,\mathrm{kHz}$    
# and a measurement time of $T_M = 64\,\mathrm{ms}$.   
# The rectangular window results from the finite measurement time $T_M$.

# ## Part I:
# At first, the frequency is given as $f_1 = \frac{4}{T_M}$.

# In[ ]:


f_rep = # TODO  Hz (representation rate for pseudo continuous-time signal)
T_M = # TODO s (measurement time)

# distance between representation points
delta_t  = 1/f_rep

# Number of points in measured signal
M = int(np.round(T_M*f_rep))

# time axis
t = np.arange(M)*delta_t # s

# sampling rate for discrete-time signal
f_s = # TODO Hz


# 1) Define the signal $x_1(t)$ here:

# In[ ]:


f_1 = 4/T_M

# TODO


# Now acquire the time-discrete signal 
# 
# \begin{align}
#     x_i(n) = x_i(n \cdot T_\mathrm{s}) = \left. x_i(t)\right|_{t = n \cdot T_\mathrm{s}}
# \end{align}
# 
# by sampling with the sampling rate $f_\mathrm{s} = 500 \mathrm{Hz}$.
# 
# To solve this, first find out the step size $N_\mathrm{s}$ between points taken from $x_1(m \cdot \Delta_t)$.

# In[ ]:


N_s = # TODO  step size between samples in x_1_t
N = # TODO Number of samples taken in total

x_1_n = # TODO

assert len(x_1_n) == N


# Plot $x_1(t)$ and $x_1(n \cdot T_s)$ together in a single figure:

# In[ ]:


def plot_time_domain(x_t, x_n,
                     y_label=(r'$x(t),\quad x(n \cdot T_\mathrm{s})$'),
                     name='Signal'):
    t_samp = np.arange(len(x_n))*1/f_s
    
    plt.figure(figsize=(20, 5))
    label_size, title_size  = 20, 25

    plt.plot(t, x_t, lw=4, color='C0')
    plt.plot(t_samp, x_n, 'o', ms= 10, color='C1')
    plt.vlines(t_samp, 0, x_n, color='C1')

    plt.xlim((t[0], t[-1]))
    plt.xlabel(r'$t / \mathrm{s}$', fontsize=label_size);
    plt.ylabel(y_label, fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=label_size)
    plt.title(name, fontsize=title_size)
    
    plt.axhline(c='k', alpha=0.75)
    plt.grid(True)


# In[ ]:


plot_time_domain(..., 
                 # y_label=r'$x_1(t),\quad x_1(n \cdot T_\mathrm{s})$', 
                 # name=r'Signal for $f_1$') 


# 2) Calculate the magnitude of $\mathrm{FFT}_M\left\{x_{1}(n)\right\}$ and $\mathrm{FFT}_N\left\{x_{1}(n)\right\}$:

# In[ ]:


X_1_M_abs = # TODO
X_1_N_abs = # TODO


# Plot the spectra together in one diagram:

# In[ ]:


def plot_spectral_domain(X_M_abs, X_N_abs, 
                         y_label=(r'$|X(\mathrm{e}^{j\theta})|,'
                                   '|X(\mathrm{e}^{j\theta_k})|$'),
                         name='Spectrum'):
    plt.figure(figsize=(20, 5))
    label_size, title_size  = 20, 25

    theta = 2*np.pi*np.arange(M)/M
    N_s = M//len(X_N_abs)
    theta_k = theta[::N_s]

    plt.plot(theta/(2*np.pi), X_M_abs, lw=4, alpha=0.5)
    plt.plot(theta_k/(2*np.pi), X_N_abs, 'o', ms= 10, color='C1')
    plt.vlines(theta_k/(2*np.pi), 0, X_N_abs, lw=2.5, color='C1')

    plt.xlim([0, 1])

    plt.xlabel(r'$\theta / 2 \pi$', fontsize=label_size)
    plt.ylabel(y_label, fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=label_size)
    plt.title(name, fontsize=title_size)

    plt.axhline(c='k', alpha=0.75)
    plt.grid(True)


# In[ ]:


plot_spectral_domain(..., 
                     # y_label=(r'$|X_1(\mathrm{e}^{j\theta})|,' 
                              # r'|X_1(\mathrm{e}^{j\theta_k})|$'), 
                     # name=r'Spectrum for $f_1$') 


# ## Part II:
# Now look at $x_2(t)$ for $f_2 = \frac{4.372}{T_M}$ 

# 1) Define the signal $x_2(t)$ here:

# In[ ]:


f_2 = 4.2/T_M
# TODO


# Acquire the time-discrete signal $x_2(n)$ by means of sampling as above:

# In[ ]:


# TODO


# Plot both time-domain signals in one figure:

# In[ ]:


plot_time_domain(...,
                 # y_label=r'$x_2(t),\quad x_2(n \cdot T_\mathrm{s})$', 
                 # name=r'Signal for $f_2$') 


# 2) Calculate the magnitude of $\mathrm{FFT}_M\left\{x_{2}(n)\right\}$ and $\mathrm{FFT}_N\left\{x_{2}(n)\right\}$;

# In[ ]:


X_2_M_abs = # TODO
X_2_N_abs = # TODO


# Plot the spectra together in one diagram:

# In[ ]:


plot_spectral_domain(..., 
                     # y_label=(r'$|X_2(\mathrm{e}^{j\theta})|,' 
                              # r'|X_2(\mathrm{e}^{j\theta_k})|$'), 
                     # name=r'Spectrum for $f_2$') 


# # Task 3
# Double the length of $x_2(nT)$ by apending zeros. What can you observe in the frequency domain?

# In[ ]:


X_2_N_abs_pad = np.abs(fft(x_2_n, 2*N))


# In[ ]:


plot_spectral_domain(..., 
                     # y_label=(r'$|X_2(\mathrm{e}^{j\theta})|,' 
                              # r'|X_{2,\mathrm{pad}}(\mathrm{e}^{j\theta_k})|$'), 
                     # name=r'Spectrum for $f_2$ with zero padding') 


# # Task 4:
# Compare the effect of different window functions on the result for $X_2(k)$. You can use the functions from `scipy.signal`.

# In[ ]:


@interact(window_name=['boxcar', 'triang', 
                  'blackman', 'hamming',
                  'hann', 'bartlett'], pad_exp=(0, 3))
def plot_windowed_spectrum(window_name, pad_exp):
    # TODO
    plt.show();

