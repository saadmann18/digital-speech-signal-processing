from .functions import *

import numpy as np
import re
import sys, traceback

from matplotlib import pyplot as plt

_EPS = np.finfo('float64').eps

class S3b:
    def __init__(self, ac, sigma, loops, mu, p5):
        mu = re.findall('(["\d+\.\d+"]+)', mu)
        for idx in range(len(mu)):
            mu[idx] = float(mu[idx])
        p5 = re.findall('([0-9]+)', p5)
        for idx in range(len(p5)):
            p5[idx] = int(p5[idx])

        # Überprüfung des Inputs
        assert len(mu) == 1 or len(p5) == 1, (
            '\nWrong Input for N and \u03BC:'
            ' Only one of these parameters is allowed to be a list. \n'
            'Example inputs: \n    \u03BC = 0.5     N = 2   4   8  16  \n'
            'or \n    \u03BC = 0.1 0.2 0.5     N = 2')

        # Überprüfen, welcher Wert als Liste übergeben wurde
        if len(mu) > 1:
            list = 'list \u03BC'
        else:
            list = 'list N'

        ac = np.minimum(ac, 99.999)
        self.sigma2Xcos = ac / (100 - ac)
        self.sigma2N = 10 ** (sigma / 10)
        self.sn2 = self.sigma2N
        self.loops = loops

        if list == 'list \u03BC':
            self.Ns = p5[0]
            self.mus = np.sort(mu)
            self.listle = len(self.mus)  # Länge Liste mu
            self.Ns = np.ones([self.listle]) * self.Ns  # mu-Liste wurde eigegeben

        elif list == 'list N':
            self.mus = mu[0]
            self.Ns = np.sort(p5)
            self.listle = len(self.Ns)  # Länge Liste N
            self.mus = np.ones([self.listle]) * self.mus  # N-Liste wurde eigegeben

        self.Ns2 = self.Ns

        # Signallängen
        self.len1 = 1000  # Länge der Signale
        if list == 'list N':
            self.len2 = int(self.len1 - self.Ns[-1] + 1)  # Plotlänge
        else:
            self.len2 = int(self.len1 - self.mus[-1] + 1)

        self.len3 = 200  # Mittelungslänge für self.Jinf

        self.sigma2Xnoise = 1  # Varianz von x[.]: Rauschanteil
        self.alphaXcos = 2 * np.pi / 10  # Normierte Frequenz des Cosinus

        self.N_stat = 2048  # Parameter für calc_x.m
        self.a = np.array([1, 0.4])  # Filterkoeffizienten des Systems
        self.b = np.array([1])

    def calc_filter_coefficients(self):
        # Ensemblemittelwert des quadr. Fehlers
        self.J = np.zeros([self.listle, self.len1 - 1])
        self.Jinf = np.zeros([self.listle])  # Abschätzung von K[ke]

        for j in range(self.listle):
            self.N = int(self.Ns[j])
            self.mu = self.mus[j]
            e2 = []  # Fehlerquadrat

            w_start = np.zeros([self.N])
            self.WW = np.zeros([self.N, self.len1])

            for idx in range(self.loops):
                x, d, self.ws, normx = signal(
                    self.len1, self.sigma2Xnoise, self.sigma2Xcos,
                    self.alphaXcos, self.sigma2N, self.b, self.a)
                e, self.W, self.w, DD = lms(self.mu, self.N, x, d, w_start)

                # Fehlerquadrat
                e2.append(
                    e[(self.N - 1):self.len1].T * e[(self.N - 1):self.len1])
                # Addition der Filterkoeffizienten
                self.WW = self.WW + self.W[:, :self.len1]

            e2 = np.asarray(e2)
            width, heigth = np.shape(e2)
            # Ensemblemittelwert des quadr. Fehlers
            self.J[j, : heigth] = np.mean(e2.T, axis=1)
            # Abschätzung von K[ke]
            self.Jinf[j] = np.mean(self.J[j, self.len2 - self.len3 : self.len2])

        # Ensemblemittelwert der Koeffizienten
        self.WW = self.WW / self.loops

        # Auswertung
        # Berechnung der Statistik und der Wiener Lösung

        Jmin = np.zeros([len(self.Ns2)])

        for st in range(len(self.Ns2)):
            self.N = int(self.Ns2[st])

            p, R, ry, sd2, self.w0 = wiener_loesung(
                self.N, self.N_stat, self.sigma2Xnoise,
                self.sigma2Xcos, self.alphaXcos, self.ws, self.sn2)

            Jmin[st] = np.abs(ry[0] - p.T @ self.w0 + self.sn2)


        # Konditionzahl
        Kond = np.linalg.cond(R)

        # Abschnätzung von J[ke], LMS
        if np.sum((self.Jinf > 10 * sd2).astype(int)) > 0:
            self.Jinf = sd2 * np.ones([self.listle])

    def plot_results(self, fig_width = 16, fig_height = 10):
        plt.close('all')

        # Plot LMS quadrierter Adaptioself.Nsfehler
        plt.figure(figsize=(fig_width, fig_height))
        plt.subplots_adjust(wspace=0.2, hspace=0.5)

        # Plot LMS Lernkurven
        length, width = np.shape(self.J[:, :-int(self.Ns[-1])].T)
        plt.subplot(325)
        plt.plot(np.log10(
            self.J[:, :-int(self.Ns[-1])].T+_EPS*np.ones([length, width])))

        plt.xlabel('$\kappa$')
        plt.ylabel('$J^{\kappa}(\mathbf{w}) / \mathrm{dB}$')
        plt.grid(True)

        if list == 'list N':
            plt.title('LMS Learning Curves in Dependence of N')
            plt.legend(self.Ns)
        else:
            plt.title('LMS Learning Curves in Dependence of $\mu$')
            plt.legend(self.mus)

        # Plot zeitlicher Verlauf der Filterkoeffizienten
        plt.subplot(321)
        plt.plot(self.W[:, self.N:self.len1 - 1].T)
        plt.xlabel('k')
        plt.ylabel('$w_1,w_2$')
        plt.title('LMS: Adaptation of the N=' f'{self.Ns[-1]}'
                  ' Weights $w_1, w_2$   (LMS Nr. ' f'{self.loops})')
        plt.xlim(0, self.len2 - 1)
        plt.ylim(-1.5, 1.5)
        plt.grid(True)

        plt.subplot(323)
        plt.plot(self.WW[:, self.N:self.len1 - 1].T)
        plt.xlabel('k')
        plt.ylabel('$w_1,w_2$')
        plt.title('LMS: Adaptation of the N=' f'{self.Ns[-1]}'
                  ' Weights $w_1, w_2$   (Enseble Average E[ w ])')
        plt.xlim(0, self.len2 - 1)
        plt.ylim(-1.5, 1.5)
        plt.grid(True)

        # Plot J[ke] und Jmin in Abhaengigkeit von N, falls N ausgewählt
        if list == 'list N':
            plt.subplot(326)
            plot1, = plt.plot(self.Ns, np.log10(self.Jinf), color='blue')
            plot2, = plt.plot(self.Ns, np.log10(Jmin), color='red')
            plt.plot(self.Ns, np.log10(self.Jinf), color='blue', marker='*')
            plt.plot(self.Ns, np.log10(Jmin), color='red', marker='*')
            plot3 = plt.fill_between(self.Ns, np.log10(Jmin), np.log10(self.Jinf),
                                     color='purple', alpha=.4)
            plt.title('$J[k_e]$ in Depencence of N')
            plt.xlabel('N')
            plt.ylabel('$J[k_e]$ / dB')

            plt.legend((plot3, plot1, plot2),
                       ('$J[k_e]$', '$J_{min}$', '$J_{inf}$'))
            plt.grid(True)

        # Plot Vergleich der Filtergewichte mit der Systemantwort
        dw1 = self.ws - np.concatenate([self.w0, np.zeros(len(self.ws) - self.N)])
        dw2 = self.ws - np.concatenate([self.w, np.zeros(len(self.ws) - self.N)])

        plt.subplot(322)
        plt.plot(self.ws, color='blue')
        plt.plot(self.w0, color='green')
        plt.plot(self.w, color='red')
        plt.title('Comparison of Filter to System Impulse Response')
        plt.xlabel('Index Coefficients')
        plt.ylabel('w')
        plt.fill_between([0, 0, self.N, self.N, 1],
                         [1.2 * np.min(self.ws), 1.2 * np.max(self.ws),
                          1.2 * np.max(self.ws), 1.2 * np.min(self.ws),
                          1.2 * np.min(self.ws)],
                         color='blue', alpha=.3)
        plt.xlim(0, len(dw1))
        plt.ylim(1.2 * np.min(self.ws), 1.2*np.max(self.ws))
        plt.legend(['System', 'Wiener Solution', 'LMS'])
        plt.grid(True)

        plt.subplot(324)
        plt.plot(dw1, color='green')
        plt.plot(dw2, color='red')
        plt.title('Difference of Filter and System Coefficients')
        plt.xlabel('Coefficient Index')
        plt.ylabel('$\Delta$ w')
        plt.fill_between([0, 0, self.N, self.N, 0],
                         [1.2*np.min(dw2), 1.2*np.max(dw2),
                          1.2*np.max(dw2), 1.2*np.min(dw2),
                          1.2*np.min(dw2)],
                         color='blue', alpha=.3)
        plt.xlim(0, len(dw1))
        plt.ylim(1.2 * np.min(dw2), 1.2*np.max(dw2))
        plt.legend(['Wiener Solution', 'LMS'])
        plt.grid(True)
