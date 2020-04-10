#!/usr/bin/env python
# coding: utf-8

# # Exercise 1


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# In[3]:

f_disp = 16000
T_m = 1000/f_disp
alpha = 25/T_m

t = np.r_[-T_m/2:T_m/2+1/f_disp:1/f_disp]
x_t = np.exp(-alpha*np.abs(t))

omega = 2*np.pi*np.r_[-f_disp/10:f_disp/10:0.1]
X_w = 2*alpha/(alpha**2 + omega**2)

# In[4]:
sample_step=12

T_s = sample_step/f_disp
n = np.round(t[::sample_step]/T_s).astype(np.int32)
x_n = np.exp(-alpha*T_s*np.abs(n))

TX_d_w = T_s*np.sinh(alpha*T_s)/(np.cosh(alpha*T_s) - np.cos(omega*T_s))

plt.figure()
plt.subplot(211)
plt.axvline(c='C7')
_, _, baseline = plt.stem(n*T_s, x_n, linefmt='C1-', markerfmt='C1o',
                          label=r'$x(n)$')
plt.setp(baseline, visible=False)
plt.axhline(c='C7')
plt.plot(t, x_t, lw=3, label=r'$x(t)$')
plt.grid(True)
plt.xlim((np.min(t), np.max(t)))
plt.yticks(ticks=(0, 0.5, 1))
plt.gca().tick_params(axis='both', which='major', labelsize=15)
plt.legend(loc='upper right', fontsize=15)

plt.subplot(212)
plt.axvline(c='C7')

plt.plot(omega*T_s/np.pi, X_w, lw=3, label=r'$X(j\omega)$')
plt.plot(omega*T_s/np.pi, TX_d_w, lw=3,
         label=r'$T \cdot X\left(\mathrm{e}^{j\omega T}\right)$')

plt.xlabel(r'$\omega\ /\ \pi / T$', fontsize=15)
plt.grid(True)
plt.xlim((np.min(omega*T_s/np.pi), np.max(omega*T_s/np.pi)))
plt.ylim((0, 3/alpha))
plt.yticks(ticks=(0, 2/alpha), labels=('0', r'$2\ /\ \alpha$'))
plt.gca().tick_params(axis='both', which='major', labelsize=15)
plt.legend(loc='upper right', fontsize=15)


# In[ ]:




