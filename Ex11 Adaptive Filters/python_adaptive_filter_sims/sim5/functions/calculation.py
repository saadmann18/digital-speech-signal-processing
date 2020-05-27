from .functions import *

import math
import numpy as np
import pandas as pd
from pandas import DataFrame

from IPython.display import display
from matplotlib import pyplot as plt
from tqdm import tqdm

class Sim5():
    """
    Simulation 5
    """
    def __init__(self, alpha_flms, alpha_lms, C, L, N,
                 N_realisierungen, sigma_n, w_system):
        cl_flag = True
        self.ar_flag = False

        self.sigma_2n = 10 ** (sigma_n / 10)
        self.sn2 = self.sigma_2n
        self.loops = N_realisierungen
        alpha = alpha_lms
        self.N = N
        self.alpha_flms = alpha_flms

        # FLMS Parameter
        if cl_flag:
            self.L = math.ceil(self.N/3)
            self.C = int(2**math.ceil(np.log2(self.L+self.N-1)))  # DFT Länge
            self.L = self.C-self.N+1  # Blockverschiebung

        self.N_FLMS = C-L+1  # Anzahl Gewichte FLMS
        self.mu_FLMS = 0.8 * self.alpha_flms  # Schrittweite FLMS
        begin = math.ceil(self.C/self.L)-1  # erster Blockindex
        self.adaptstart = (begin+1)*self.L

        # Signallängen
        self.len1 = 24000
        self.len2 = int(self.len1 - self.adaptstart - self.L + 1)  # Plotlänge
        self.len3 = 2000  # Mittelungslänge für self.Jinf

        # Schrittweite
        self.mu = alpha * 0.8 * 1 / self.N  # angepasst an N

        self.N_stat = 4096  # Parameter für calc_x.m

        # laden der Systemantwort
        # w_system FIR
        if w_system == '1500':
            self.b = np.asarray(pd.read_csv('coefficients/w_system_1500.csv'))[:, 0]
        else:
            self.b = np.asarray(pd.read_csv('coefficients/w_system_700.csv'))[:, 0]
        self.a = np.array([1])

        # Laden der AR-Koeffizienten 16. Ordnung
        self.ar_koeff = np.asarray(pd.read_csv('coefficients/ar_koff16.csv'))[0, :]

    def calc_filter_coefficients(self):
        e2_LMS = np.zeros([self.len2, self.loops])  # Fehlerquadrat LMS
        e2_FLMS = np.zeros([self.len2, self.loops])  # Fehlerquadrat FLMS
        w_start = np.zeros([self.N])  # Startwerte Filterkoeffizienten

        for idx in tqdm(range(self.loops)):
            # Generierung der Signale
            x, d, self.ws, norm = signal_ar(self.len1, self.sigma_2n, self.b, self.a, self.ar_flag, self.ar_koeff)

            # FLMS Algorithmus
            e_FLMS, self.w_FLMS, d_hat_FLMS, self.calc_FLMS, self.p_hat_FLMS = flms(self.mu_FLMS, self.C, self.L, x, d)

            # Fehlerquadrat FLMS
            e2_FLMS[:, idx] = e_FLMS[self.adaptstart: self.len1-self.L+1]**2

            # LMS Algorithmus
            e_LMS, self.w_LMS, d_hat_LMS, self.calc_LMS = lms(self.mu, self.N, x, d, w_start, self.adaptstart)

            # Fehlerquadrat LMS
            e2_LMS[:, idx] = e_LMS[self.adaptstart: self.len1-self.L+1]**2

        self.idx = idx
        # Ensemblemittelwert
        self.J_FLMS = np.mean(e2_FLMS, axis=1)
        self.J_LMS = np.mean(e2_LMS, axis=1)

        # zusätzliche zeitliche Glättung der Lernkurven mit einem exponentiellen Fenster aus Darstellungsgründen

        self.J_FLMS = expwin2(self.J_FLMS)
        self.J_LMS = expwin2(self.J_LMS)

        # Berechnung der Statistik und self.Jmin

        # Eingang AR-Prozess
        if self.ar_flag:
            self.rx = calc_R2(np.array([1]), self.ar_koeff, self.N, self.N_stat)

            ryx = calc_ryx_freq(np.transpose(self.ws), np.array([1]), self.rx, self.N_stat)
            p = np.transpose(ryx[0:self.N])  # Kreuzkorrelationsvektor p
            ry = calc_ry_freq(np.transpose(self.ws), np.array([1]), self.rx, self.N_stat)  # Autokorrelations Funktion von d'
            self.sd2 = ry[0] + self.sigma_2n  # Varianz von d

            self.Jmin = np.abs(self.sd2 -np.transpose(p) @ self.ws[0:self.N])

            # Um w0 zu berechnen, muesste inv(R) bestimmt werden. Dies
            # ist bei NxN=1000x1000 zu aufwendig, so dass hier
            # näherungsweise w0=self.ws(1:N) gesetzt wird.
        else:
            self.rx = np.zeros(self.N_stat)
            self.rx[0] = 1
            self.Jmin = np.abs((self.sn2 + np.sum(self.ws[self.N: len(self.ws)])**2))
            self.sd2 = 1 + self.sigma_2n

        # Grenzwerte der Lernkurven, Fehleinstellung, ERLE
        # Abschätzung von J[ke], LMS
        self.Jinf = np.mean(self.J_LMS[self.len2-self.len3: self.len2])

        if self.Jinf > 10 * self.sd2 or math.isnan(self.Jinf):
            #  keine Konvergenz, Schrittweite mu zu groß
            self.Jinf = self.sd2

        # Abschätzung von J[ke], FLMS
        self.Jinf2 = np.mean(self.J_FLMS[self.len2 - self.len3: self.len2])

        # Fehleinstellung von LMS
        M = (self.Jinf - self.Jmin) / self.Jmin
        if M > 10:
            self.Mplot = ' > 1000%'
        else:
            self.Mplot = str(np.round(100*M, 2)) + '%'

        # Fehleinstellung von FLMS
        M2 = (self.Jinf2 - self.Jmin) / self.Jmin
        if M2 > 10:
            self.M2plot = ' > 1000%'
        else:
            self.M2plot = str(np.round(100*M2, 2)) + '%'

        # ERLE
        erle_LMS = 10 * np.log10(self.sd2/self.Jinf)
        erle_FLMS = np.round(10 * np.log10(self.sd2/self.Jinf2), 2)
        self.erle_FLMS_db_string = str(erle_FLMS) + 'dB'

        # Plotparameter
        if self.Jinf <= self.Jmin:
            self.Jinf = self.Jmin
        if self.Jinf2 <= self.Jmin:
            self.Jinf2 = self.Jmin
        self.sn2_db = 10 * np.log10(self.sn2)
        self.Jmin_db = 10 * np.log10(self.Jmin)

        # Berechnung des Systemfehlermaßes
        # Im Zeitbereich
        self.w_FLMS_plot = np.concatenate([self.w_FLMS, np.zeros(len(self.ws)-self.N)])
        self.dw_2 = self.ws - np.concatenate([np.array([0]), self.w_LMS[0:-1], np.zeros(len(self.ws)-self.N)])  # self.w_LMS ist um ein sample nach links verschoben
        self.dw_3 = self.ws - self.w_FLMS_plot[0:len(self.ws)]

        self.dw_2_db = 10 * np.log10(np.sum(self.dw_2**2) / np.sum(self.ws**2))
        self.dw_3_db = 10 * np.log10(np.sum(self.dw_3**2)) / np.sum(self.ws**2)

        # im Frequenzvereich
        self.ws_freq = np.abs(np.fft.fft(self.ws, 8000))
        self.DW2_db = 10 * np.log10((np.abs(np.fft.fft(self.dw_2, 8000))**2) / self.ws_freq**2)
        self.DW3_db = 10 * np.log10((np.abs(np.fft.fft(self.dw_3, 8000))**2) / self.ws_freq**2)

    def show_results(self):
        # Tabelle
        print('Simulation finished')
        values_dict = {
            'Realization #': int(self.idx+1),
            '$N_{LMS}$=N': int(self.N),
            '$N_{FLMS}$=C-L+1': self.N_FLMS,
            '# Blocks': math.floor(self.len1/self.L),
            'Computation time ratio LMS/FLMS': np.round(self.calc_LMS/self.calc_FLMS),
        }

        df = DataFrame.from_dict(values_dict, orient='index', columns=[''])
        display(df)

        # Plots
        plt.close('all')

        # Hilfsparameter
        # y-Positionen
        axis_1 = 10 * np.log10(self.Jmin/10)  # ax1(3)
        axis_2 = 10 * np.log10(self.sd2 * 10)  # ax1(4)
        y_range = axis_2 - axis_1

        # Plot Parameter
        fig_width = 16
        fig_height = 14

        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.subplots_adjust(top=0.95, wspace=0.3, hspace=0.7)

        # AKF und Spektrum von X
        plt.subplot(521)
        rxbs = np.concatenate([np.flip(self.rx[2: len(self.rx)]), self.rx])
        x_axis = np.linspace(-200, 200, len(rxbs[self.N_stat-200:self.N_stat+200]))
        plt.plot(x_axis, rxbs[self.N_stat-200:self.N_stat+200])
        plt.xlim((-200, 200))
        plt.ylim((-1, 1.2))
        plt.grid(True)
        plt.title('ACF of x')
        plt.xlabel('$\lambda$')
        plt.ylabel('$\phi_{xx}$')

        plt.subplot(522)
        px = np.abs(np.fft.fft(rxbs, 8000))
        plt.plot(10*np.log10(px[0:4000]))
        plt.xlim((0, 4000))
        plt.ylim((-30, 20))
        plt.grid(True)
        plt.title('Power Spectral Density of x')
        plt.xlabel('$f$ / Hz')
        plt.ylabel('$\Phi_{xx}$ / dB')

        # Lernkurven
        # LMS Lernkurven
        ax1 = plt.subplot(523)
        plt.plot(10*np.log10(self.J_LMS))
        plt.xlim((0, self.len2))
        plt.ylim((10*np.log10(self.Jmin/10), 10*np.log10(self.sd2*10)))
        plt.grid(True)
        plt.title('LMS: Learning Curve J[k] = $E[e[n]^2]$'
                  f'Estimate using {self.loops} Realizations')
        plt.xlabel('k')
        plt.ylabel('$J$ / dB')

        y_pos_1 = np.max([10*np.log10(self.Jmin) + 0.06 * y_range,
                          10 * np.log10(self.Jinf)]) * np.ones(len(self.J_LMS))
        y_pos_2 = np.max([10*np.log10(self.Jmin) + 0.01 * y_range,
                          10 * np.log10(self.Jinf)]) * np.ones(len(self.J_LMS))

        plt.plot(y_pos_2, 'g')
        plt.plot(10*np.log10(self.sd2 * np.ones(self.len2)), 'r')
        plt.plot(10*np.log10(self.Jmin * np.ones(self.len2)), 'r')

        # Kommentare im Plot
        plt.text(-self.len2*0.15, y_pos_1[0], '$J[k_e]$')
        plt.text(-self.len2*0.15, 10*np.log10(self.Jmin), '$J_{min}$')
        plt.text(1.01 * self.len2, y_pos_1[0],
                 f'{np.round(10*np.log10(self.Jinf), decimals=2)} dB')
        plt.text(-self.len2 * 0.15, 10*np.log10(self.sd2), 'J[0]=$\sigma_d^2$')
        plt.text(1.01 * self.len2, 10 * np.log10(self.sd2),
                 f'{np.round(10*np.log10(self.sd2), decimals=2)} dB')
        plt.text(1.01 * self.len2, 10 * np.log10(self.Jmin),
                 f'{np.round(10*np.log10(self.Jmin), decimals=2)} dB')

        # RLS-Lernkurven
        plt.subplot(524)
        plt.plot(10*np.log10(self.J_FLMS))
        plt.xlim((0, self.len2))
        plt.ylim((10*np.log10(self.Jmin/10), 10*np.log10(self.sd2*10)))
        plt.grid(True)
        plt.title('FLMS: Learning CurveJ[k] = $E[e[k]]^2$'
                  f'Estimate using {self.loops} Realizations')
        plt.xlabel('k')
        plt.ylabel('$J$ / dB')

        y_pos_1 = np.max([10*np.log10(self.Jmin) + 0.06 * y_range,
                          10 * np.log10(self.Jinf2)]) * np.ones(len(self.J_LMS))
        y_pos_2 = np.max([10*np.log10(self.Jmin) + 0.01 * y_range,
                          10 * np.log10(self.Jinf2)]) * np.ones(len(self.J_LMS))

        plt.plot(y_pos_2, 'g')
        plt.plot(10*np.log10(self.sd2 * np.ones(self.len2)), 'r')
        plt.plot(10*np.log10(self.Jmin * np.ones(self.len2)), 'r')

        # Kommentare im Plot
        plt.text(-self.len2*0.15, y_pos_1[0], '$J[k_e]$')
        plt.text(-self.len2*0.15, 10*np.log10(self.Jmin), '$J_{min}$')
        plt.text(1.01 * self.len2, y_pos_1[0],
                 f'{np.round(10*np.log10(self.Jinf2), decimals=2)} dB')
        plt.text(-self.len2 * 0.15, 10*np.log10(self.sd2), 'J[0]=$\sigma_d^2$')
        plt.text(1.01 * self.len2, 10 * np.log10(self.sd2),
                 f'{np.round(10*np.log10(self.sd2), decimals=2)} dB')
        plt.text(1.01 * self.len2, 10 * np.log10(self.Jmin),
                 f'{np.round(10*np.log10(self.Jmin), decimals=2)} dB')

        text = (
            '$J_{min}$ ' f'(for N= {self.N}) = '
            f'{np.round(self.Jmin_db, 2)} dB \t'
            '$J_{ex} = J[k_e] - J_{min}$ \t'
            '$M_{LMS} = J_{ex}/J_{min} = $'
            f'{self.Mplot} \t'
            '$M_{FLMS} = J_{ex}/J_{min}$ = '
            f'{self.M2plot} \t'
            '$ERLE_{FLMS} = \sigma_d^2 / J[k_e] = $'
            f'{self.erle_FLMS_db_string}'
        )

        ax1.text(0, -0.42, text, size=9,
                 bbox=dict(facecolor='gray', alpha=0.5),
                 transform=ax1.transAxes)

        # Vergleich der Filtergewichte mit der Systemantwort
        #Vergleich der Systemantwort
        ax2 = plt.subplot(525)
        plt.plot(self.ws, 'b')
        plt.plot(self.w_LMS, 'r')
        plt.plot(self.w_FLMS, 'g')
        plt.xlim(0, len(self.ws) + 50)
        plt.ylim(1.2*np.min([self.ws]), 1.2*np.max([self.ws]))

        plt.title('Comparison of filter and system response')
        plt.xlabel('Coefficient Index')
        plt.ylabel('w')
        plt.legend(('System', 'LMS', 'FLMS'), loc='upper right')
        plt.fill_between([0, 0, self.N, self.N, 0],
                         [1.2*np.min([self.ws]),
                          1.2*np.max([self.ws]),
                          1.2*np.max([self.ws]),
                          1.2*np.min([self.ws]),
                          1.2*np.min([self.ws])],
                         color='blue', alpha=.3)

        # Abweichung von der Systemantwort
        plt.subplot(526)
        plt.plot(self.dw_2, 'r')
        plt.plot(self.dw_3, 'g')
        plt.title('Difference of Filter and System coefficients')
        plt.xlim(0, len(self.ws)+50)
        plt.ylim(1.2*np.min((self.dw_2, self.dw_3)),
                 1.2*np.max((self.dw_2, self.dw_3)))
        plt.xlabel('Coefficient Index')
        plt.ylabel('$\u0394$ w')
        plt.legend(('LMS', 'FLMS'),loc='upper right')
        plt.grid(True)

        text = ('$\Delta w_{dB (LMS)}   = $'
                + str(np.round(self.dw_2_db)) + 'dB \t'
                +'$\Delta w_{dB (FLMS)}   = $'
                + str(np.round(self.dw_3_db)) + 'dB'
        )

        ax2.text(0, -0.42, text, size=9,
                 bbox=dict(facecolor='gray', alpha=0.5),
                 transform=ax2.transAxes)

        plt.subplot(527)
        self.w_FLMS_plot = np.concatenate(
            (self.w_FLMS, np.zeros([len(self.ws)-self.N])))
        plt.plot(10*np.log10(self.ws_freq), 'b')
        plt.plot(10*np.log10(np.abs(np.fft.fft(self.w_FLMS_plot[0:len(self.ws)], n=8000))),
                 'g')
        plt.plot(10*np.log10(np.abs(np.fft.fft(self.w_LMS, n=8000))), 'r')
        plt.grid(True)
        plt.xlim(0, 4000)
        plt.ylim(-30, 10)
        plt.title('Filter in Frequency Domain :'
                  '$W_{System}$, $w_{LMS}$ and $W_{FLMS}$')
        plt.xlabel('$f$ / Hz')
        plt.ylabel('$W$ / dB')
        plt.legend(('$W_{System}$', '$W_{LMS}$', '$W_{FLMS}$'),
                   loc='upper right')

        # Abweichung von der Systemantwort (Systemfehlermaß)
        plt.subplot(528)
        plt.plot(self.DW2_db, 'r')
        plt.plot(self.DW3_db, 'g')
        plt.grid(True)
        plt.title('System Error Measure in Frequency Domain: $\u0394_{dB}$')
        plt.xlabel('$f$ / Hz')
        plt.ylabel('$\u0394 W_{dB}$ / dB')
        plt.legend(('LMS', 'FLMS'), loc='upper right')

        # Frequenzabhaengige  Schrittweite mu(j) = alphaFLMS / PX(j)
        plt.subplot(529)
        plt.plot(-10*np.log10(self.p_hat_FLMS[0:int(self.C/2)]/self.alpha_flms)
                 + 10 * np.log10(self.C))
        plt.title('FLMS-Algorithm: Frequency-dependent Step Width '
                  '$\u03BC _j$ = $\u03B1 _{FLMS}$ / $Px_j$')
        plt.xlabel('j(DFT Coeffificent)C/2')
        plt.ylabel('$\u03BC_j$ / dB')
        plt.xlim(0, self.C/2)
        plt.ylim(-10, 10)
        plt.grid(True)

        plt.show()
