from .functions import *

import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class S3a:
    def __init__(self, ac, p2, loops, mu):
        self.loops=loops
        self.mu = mu
        self.sigma2Xcos = ac/(100-ac)
        self.sigma2N = 10 ** (p2/10)
        self.sn2 = self.sigma2N

        # Number of filter coefficients
        self.N = 2

        # Signal lengths
        self.len1 = 800  # Signal length
        self.len2 = self.len1 - self.N  # Plot length
        self.len3 = 200  # Averaging length for self.Jinf

        # Parameter input signal
        self.sigma2Xnoise = 1  # Variance of x[.]: Noise component
        self.alphaXcos = 2 * np.pi / 15  # Normalized frequency of the cosine

        self.Nstep = 300  # Number of steps Newton / Gradient-descent
        self.cNewton = 0.05  # Increment Newton-descent
        self.cGrad = 0.05  # Increment Grad-Verfahren

        # Range of filter coefficients for the area display error
        start = -1.5
        end = 0.1
        step = 0.025
        self.w2_range = np.linspace(start, end, int(abs((start-end)/step))+1)
        start = -1.1
        end = 0.5
        step = 0.025
        self.w1_range = np.linspace(start, end, int(abs((start-end)/step))+1)

        # System filter coefficients

        self.b = np.array([0.15, -1])
        self.a = np.array([1])

        self.N_stat = 2048
        self.w_start = np.zeros([self.N])  # Initial values of filter coefficients


    def calc_filter_coefficients(self):
        self.w_2 = np.zeros([self.N, self.len1])
        self.e_2 = np.zeros([self.len2, self.loops])
        for self.idx in range(self.loops):
            x, d, ws, normx = signal(self.len1, self.sigma2Xnoise,
                                     self.sigma2Xcos,self.alphaXcos,
                                     self.sigma2N, self.b, self.a)

            e, W, w, DD = lms(self.mu, self.N, x, d, self.w_start)

            # Square of error
            self.e_2[:, self.idx] = np.abs(e[self.N - 1:self.len1 - 1]) ** 2

            # Addition of the filter coefficients
            self.w_2 = self.w_2 + W[:, 0:self.len1]

        self.idx = self.idx
        self.W = W

        self.J = np.mean(self.e_2.T, axis=0)
        self.w_2 /= self.loops

        self.R, rx = calc_R(self.sigma2Xnoise, self.sigma2Xcos,
                            self.alphaXcos,self.N, self.N_stat)

        ryx = calc_ryx(ws, 1, rx, self.N_stat)
        self.p = ryx[0:self.N]

        ry = calc_ry(ws, 1, rx, self.N_stat)

        self.sd2 = ry[0] + self.sn2

        # Wiener solution
        self.w0 = np.linalg.inv(self.R) @ self.p

        # self.Jmin
        self.Jmin = np.abs(ry[0]-self.p.T @ self.w0 + self.sn2)

        # Condition number
        Kond = np.linalg.cond(self.R)

        # Newton method
        self.wNewton = newton(self.p, self.R, self.w_start,
                              self.cNewton, self.Nstep)
        self.w1Newton = self.wNewton[0, :-1]
        self.w2Newton = self.wNewton[1, :-1]

        # Gradient method
        self.wGrad = grad(self.p, self.R, self.w_start, self.cGrad, self.Nstep)

        #Estimation of J[ke], LMS
        self.Jinf = np.mean(self.J[self.len2-self.len3-1:self.len2-1])
        if self.Jinf > 10 * self.sd2 or self.Jinf is None:
            self.Jinf = self.sd2

    def plot_results(self):
        plt.close('all')

        if self.Jinf <= self.Jmin or self.Jinf is None:
            self.Jinf = self.Jmin * 1.00000001

        fig_width = 16
        fig_height = 10

        # Plot LMS squared adaptation error
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.subplots_adjust(wspace=0.3, hspace=0.5)

        yrange = np.log10(self.sd2 * 10) - np.log10(self.Jmin / 10)
        plotypos2 = max(np.log10(self.Jmin) + 0.06 * yrange, np.log10(self.Jinf));
        plotypos2b = max(np.log10(self.Jmin) + 0.01 * yrange, np.log10(self.Jinf));

        plt.subplot(421)
        plt.plot(np.log10(self.e_2[:, self.idx]))
        plt.plot(np.log10(self.sd2*np.ones([self.len2])), 'r')
        plt.plot(np.log10(self.Jmin * np.ones([self.len2])), 'r')
        plt.xlim(0, self.len2)
        plt.ylim(np.log10(self.Jmin/10), np.log10(self.sd2*10))

        plt.title('LMS: Squared Adaptation Error $e[n]^2$'
                  f' (LMS Nr. {self.idx})')
        plt.text(-self.len2*0.08, np.log10(self.sd2*10)+yrange*0.05, '$J$ / dB')
        plt.text(-self.len2*0.15, np.log10(self.Jmin), '$J_{min}$')
        plt.text(-self.len2*0.15, np.log10(self.sd2), 'J[0] = $\sigma_d^2$')
        plt.text(self.len2*1.01, np.log10(self.sd2),
                 str(np.round(10*np.log10(self.sd2), decimals=4))+' dB')
        plt.text(self.len2*1.01, np.log10(self.Jmin),
                 str(np.round(10*np.log10(self.Jmin), decimals=4))+' dB')
        plt.grid(True)

        # Plot Learning curve J[k]
        plt.subplot(423)
        plt.plot(np.log10(self.J))
        plt.plot(plotypos2b*np.ones([self.len2]), 'g')
        plt.plot(np.log10(self.sd2*np.ones([self.len2])), 'r')
        plt.plot(np.log10(self.Jmin * np.ones([self.len2])), 'r')
        plt.xlim(0, self.len2)
        plt.ylim(np.log10(self.Jmin/10), np.log10(self.sd2*10))

        plt.title('LMS:  Learning Curve J[\kappa] = $E[ e[n]^2 ]$'
                  f' ({self.loops} Realizations)')
        plt.text(-self.len2*0.08, np.log10(self.sd2*10)+yrange*0.05, '$J$ / dB')
        plt.text(-self.len2*0.15, np.log10(self.Jmin), '$J_{min}$')
        plt.text(-self.len2*0.15, np.log10(self.sd2), 'J[0] = $\sigma_d^2$')
        plt.text(self.len2*1.01, np.log10(self.sd2),
                 f'{np.round(10*np.log10(self.sd2), decimals=4)} dB')
        plt.text(self.len2*1.01, np.log10(self.Jmin),
                 f'{np.round(10*np.log10(self.Jmin), decimals=4)} dB')
        plt.text(self.len2*1.01, plotypos2,
                 f'{np.round(10*np.log10(self.Jinf), decimals=4)} dB')
        plt.grid(True)

        # Calculation of additional parameters
        Jflaeche = calc_J(self.sd2, self.p, self.R,
                          arg1=self.w1_range, arg2=self.w2_range)

        w1LMS = self.W[0, :-1]
        w2LMS = self.W[1, :-1]
        self.w_2_1LMS = self.w_2[0, :-1]
        self.w_2_2LMS = self.w_2[1, :-1]
        w1Grad = self.wGrad[0, :-1]
        w2Grad = self.wGrad[1, :-1]

        # Plot representation of the recursion course of the different
        # Adaptive algorithms on the error surface (2D plot)
        x, y = np.meshgrid(self.w1_range, self.w2_range)
        plt.subplot(222)
        plt.contour(x, y, Jflaeche,
                    np.linspace(self.Jmin, np.max(np.max(Jflaeche)), 50))

        lms_plt, = plt.plot(w1LMS, w2LMS, 'b', label='LMS')
        elms_plt, = plt.plot(self.w_2_1LMS, self.w_2_2LMS, 'r',
                             label='E[LMS] (Ensemble Average)')
        gradient_plt, = plt.plot(w1Grad, w2Grad, 'g', label='Gradient')
        newton_plt, = plt.plot(self.w1Newton, self.w2Newton, 'y',
                               label='Newton')
        plt.plot(self.w0[0], self.w0[1], marker='x', markersize=50, color='red')

        plt.xlim(self.w1_range[0], self.w1_range[len(self.w1_range) - 1])
        plt.ylim(0.8 * self.w2_range[0], self.w2_range[len(self.w2_range) - 1])

        plt.title('Recursion of Different Adaptation Algorithms\n'
                  ' (2D, Contourplot)')
        plt.xlabel('$w_1$')
        plt.ylabel('$w_2$')
        plt.legend(handles=[lms_plt, elms_plt, gradient_plt, newton_plt])
        plt.grid(True)

        # Plot of the filter coefficients over time
        plt.subplot(425)
        plt.plot(self.W[:, self.N:self.len1 - 1].T)
        plt.xlabel('\kappa')
        plt.ylabel('$w_1,w_2$')
        plt.title('LMS: Adaptation of Weights $w_1, w_2$'
                  f'(LMS Nr. {self.loops})')
        plt.xlim(0, self.len2 - 1)
        plt.ylim(-1.5, 1.5)
        plt.grid(True)

        # Plot LMS: Adaptation of the weights
        plt.subplot(427)
        plt.plot(self.w_2[:, self.N:self.len1 - 1].T)
        plt.xlabel('\kappa')
        plt.ylabel('$w_1,w_2$')
        plt.title('LMS: Adaptation of Weights $w_1, w_2$'
                  '(Ensemble Average $E[w]$)')
        plt.xlim(0, self.len2 - 1)
        plt.ylim(-1.5, 1.5)
        plt.grid(True)

        # Plot Representation of the recursion course of the different
        # Adaptive algorithms on the error surface (3D plot)
        offset = 0.0  # Distance of the recursion profiles to the defect area

        JLMS = calc_J(self.sd2, self.p, self.R, arg1=self.W)
        JELMS = calc_J(self.sd2, self.p, self.R, arg1=self.w_2)
        self.JNewton = calc_J(self.sd2, self.p, self.R, arg1=self.wNewton)
        JGrad = calc_J(self.sd2, self.p, self.R, arg1=self.wGrad)

        ax = fig.add_subplot(224, projection='3d')

        x, y = np.meshgrid(self.w1_range, self.w2_range)

        ax.plot_surface(x, y, Jflaeche, rstride=4, cstride=4, linewidth=5,
                        antialiased=True, alpha=0.4, cmap='cividis')

        JLMS = JLMS[:-1] + offset
        JELMS = JELMS[:-1] + offset
        JGrad = JGrad[:-1] + offset
        self.JNewton = self.JNewton[:-1] + offset

        lms_plt, = ax.plot(w1LMS, w2LMS, JLMS, color='red', label='LMS')
        elms_plt, = ax.plot(self.w_2_1LMS, self.w_2_2LMS, JELMS,
                            color='darkviolet',
                            label='E[LMS]  (Ensemble Average)')
        gradient_plt, = ax.plot(w1Grad, w2Grad, JGrad,
                                color='yellow', label='Gradient')
        newton_plt, = ax.plot(self.w1Newton, self.w2Newton, self.JNewton,
                              color='black', label='Newton')

        ax.set_xlabel('$w_1$')
        ax.set_ylabel('$w_2$')
        ax.set_zlabel('J')
        ax.set_xlim(self.w1_range[0], self.w1_range[-1])
        ax.set_ylim(self.w2_range[0], self.w2_range[-1])
        plt.title('Error Surface Trajectories of'
                  'Different Adaptation Algorithms', loc='left')
        ax.legend(handles=[lms_plt, elms_plt, gradient_plt, newton_plt], loc='best')
