import math
import numpy as np

from random import *
from scipy import signal as sg
from scipy.linalg import toeplitz


def calc_J(sd2, p, R, **kwargs):
    """
    Calculates the curve on the defect area or the
    Defect area itself

    @param sd2:
    @param p: Cross correlation vector
    @param R: Autocorrelation matrix
    @param kwargs:  if only arg1 specified:
                        arg1 : 2xN Matrix with different w values
                    if arg1 and arg2 are specified:
                        arg1 contains the w1 and arg2 contains the
                        w2 values for which the defect area is calculated
    @return: J - See curve on the defect area or the defect area **kwargs
    """
    if 'arg2' not in kwargs:
        w = kwargs['arg1']
        h, b = np.shape(w)
        J = np.zeros([b])

        for idx in range(b):
            J[idx] = sd2 - 2 * p.T @ w[:, idx] + w[:, idx].T @ R @ w[:, idx]

    elif 'arg2' in kwargs:
        J = np.zeros([len(kwargs['arg2']), len(kwargs['arg1'])])

        for idx_i in range(len(kwargs['arg1'])):
            for idx_j in range(len(kwargs['arg2'])):
                w = []
                w.append(kwargs['arg1'][idx_i])
                w.append(kwargs['arg2'][idx_j])
                w = np.asarray(w)

                J[idx_j, idx_i] = sd2 - 2 * p.T @ w + w.T @ R @ w

    return J

def calc_R(sigma2Xnoise, sigma2Xcos, alphaXcos, N, N_stat):
    """
    Berechnung der Autokorrelations- Funktion und Matrix
    Eingangssignal: x = cos + weisses Rauschen

    @param sigma2Xnoise: Varianz  weisses Rauschen
    @param sigma2Xcos: Varianz cos
    @param alphaXcos: Normierte Frequenz cos
    @param N: Dimension N x N von R
    @param N_stat: Länge von rx
    @return: R - Autokorrelationsmatrix, rx - Autokorrelationsfunktion
    """
    R = np.zeros([N, N])
    rn = np.zeros([N_stat])

    rcos = sigma2Xcos * np.cos(np.linspace(0, N_stat-1, N_stat) * alphaXcos)
    rn[0] = sigma2Xnoise
    rx = rcos + rn
    rx /= sigma2Xnoise + sigma2Xcos
    r = rx[0:N]
    R = toeplitz(r, r)

    return R, rx

def calc_ry(b,a,rx,N_stat):
    """
    Berechnung der Autokorrelationsfunktion ry des Ausgangs
    eines Filters

    N_stat: Faltungslaenge. Je laenger rx oder
    die Systemstossantwort, desto groesser muss N_stat
    sein, um eine gleiche Genauigkeit zu erreichen.

    Berechnung: ry[.] = h[.] * h[-.] * rx
        h: Systemstossantwort

    @param b: Filterkoeffizienten
    @param a: Filterkoeffizienten
    @param rx: Autokorrelationsfunktion x
    @param N_stat: Faltungslänge
    @return: ry - Autokorrelationsfunktion y
    """
    N_stat = 2**(math.ceil(np.log2(N_stat)))
    Nl = N_stat/2

    # Autokorrelationsfunktion von x
    rx = np.concatenate([rx, np.zeros(N_stat-len(rx))], axis=0)
    rx = np.concatenate([np.flip(rx), rx[1:]], axis=0)

    # Systemantstoßwort
    dirac = np.zeros([N_stat])
    dirac[0] = 1

    h = sg.lfilter(b, a, dirac)
    hr = np.flip(h)

    hrs = hr[N_stat-int(Nl):N_stat]
    h_hm = sg.lfilter(hrs, 1, h)

    # Kreuzkorrelationsfunktion ry
    ry1 = sg.lfilter(h_hm, 1, rx)
    ry = ry1[int(Nl) - 2 + N_stat:len(ry1)]
    
    return ry

def calc_ryx(b, a, rx, N_stat):
    """
    function [ryx]=calc_ryx(b,a,rx,N_stat);

    Berechnung der Kreuzkorrelationsfunktion ryx

    N_stat: Faltungslaenge. Je laenger rx oder
    die Systemstossantwort, desto groesser muss N_stat
    sein, um eine gleiche Genauigkeit zu erreichen.

    Berechnung: ryx[.] = h[.] * rx
        h: Systemstossantwort

    @param b: Filterkoeffizienten
    @param a: Filterkoeffizienten
    @param rx: Autokorrelationsfunktion x
    @param N_stat: Faltungslänge
    @return: ryx - Kreuzkorrelationsfunktion
    """
    N_stat = 2**(math.ceil(np.log2(N_stat)))
    Nl = N_stat/2

    # Autokorrelationsfunktion von x
    rx = np.concatenate([rx, np.zeros(N_stat-len(rx))], axis=0)
    rx =np.concatenate([np.flip(rx), rx[1:]], axis=0)

    # Systemantstoßwort
    dirac = np.zeros([N_stat])
    dirac[0] = 1

    h = sg.lfilter(b, a, dirac)

    # Kreuzkorrelationsfunktion ryx
    ryx = sg.lfilter(h, 1, rx)
    ryx = ryx[N_stat-1:len(ryx)]
    
    return ryx

def grad(p, R, w_start, cGrad, Nstep):
    """
    Gradienten-Verfahren

    @param p: Kreuzkorrelationsvektor
    @param R: Autokorrelationsmatrix
    @param w_start: Startwerte Filterkoeffizienten
    @param cGrad: Schrittweite
    @param Nstep:  Anzahl Iterationen
    @return: wGrad - Filterkoeffizienten im zeitlichen Verlauf
    """
    N = len(w_start)
    wGrad = np.zeros([N, Nstep])
    wGrad[:, 1] = w_start

    for idx in range(Nstep-1):
        Gradient = 2*(R @ wGrad[:, idx] - p)

        wGrad[:, idx+1] = wGrad[:, idx] - cGrad * Gradient

    return wGrad

def lms(mu, N, X, D, w_start):
    """
    LMS-Algortihmus

    @param mu: Schrittweite
    @param N: Anz. Filterkoeffizienten = Filterordnung+1
    @param X: Filtereingang X[.]
    @param D: erwünschtes Signal D[.]
    @param w_start: Startwerte Filterkoeffizienten
    @return: E - Fehlersignal E[.], W - Filterkoeffizienten im zeitlichen Verlauf, w - Filterkoeffizienten am Ende der Adaption, DD - Filterausgang, Schaetzung von D
    """
    adaptlen = len(X)
    w = w_start
    W = np.zeros([N, adaptlen])
    DD = np.zeros([adaptlen])
    E = np.zeros([adaptlen])

    idx = N
    while idx < adaptlen:
        W[:, idx-1] = w
        x = np.flip(X[idx-N:idx])
        y = x.T @ w
        e = D[idx-1] - y
        w = w + mu * e * x
        E[idx-1] = e
        DD[idx-1] = y

        idx += 1

    return E, W, w, DD

def newton(p, R, w_start, cNewton, Nstep):
    """
    Newton-Verfahren

    @param p: Kreuzkorrelationsvektor
    @param R: Autokorrelationsmatrix
    @param w_start: Startwerte Filterkoeffizienten
    @param cNewton: Schrittweite
    @param Nstep: Anzahl Iterationen
    @return: wNewton - Filterkoeffizienten im zeitlichen Verlauf
    """
    N = len(w_start)  # Anzahl Filterkoeffizienten des FIR-Filters

    wNewton = np.zeros([N,Nstep])  # Initialisierung
    wNewton[:,1] = w_start  # Startwert

    for idx in range(Nstep-1):
        Gradient = 2*(R @ wNewton[:,idx] - p)  # Gradientenberechnung
        wNewton[:, idx+1] = wNewton[:, idx] - cNewton * np.linalg.inv(2*R) @ Gradient

    return wNewton

def signal(length, sigma2X, sigma2Xcos, alphaXcos, sigma2N, b, a):
    """
    Erzeugung der Signale X,D
    mit X Eingangssignal = weißes Rauschen
        D gewünschte Signal = Systemausgang + Messrauschen

    @param length: Signallänge
    @param sigma2X: Varianz weißes Rauschen
    @param sigma2Xcos:  Varianz cos
    @param alphaXcos: Normierte Frequenz cos
    @param sigma2N: Varianz weißes Messrauschen
    @param b: Filterkoeffizienten des Systemfilters
    @param a: Filterkoeffizienten des Systemfilters
    @return: X - Filtereingang, D - erwünschtes Signal, ws - Systemstoßantwort (gegeben durch b und a), norm - Normierung des Systemfilters
    """
    t = np.linspace(0, length-1, length)
    X = np.sqrt(sigma2X) * np.random.randn(length) + np.sqrt(2 * sigma2Xcos) * np.cos(
        alphaXcos * t.T + 2 * np.pi * random())

    normX = np.sqrt(np.var(X))  # Normierung der Eingangsleistung auf var(X) =1
    X /= normX

    Ys = sg.lfilter(b, a, X)  # Systemausgang verändert das Signal mit einem FIR filter überprüfen??

    normYs = np.sqrt(np.var(Ys)) # Normierung der Ausgangsleistung auf var(X) =1
    Ys /= normYs

    # weißes Rauschen
    N = np.sqrt(sigma2N)*np.random.randn(length)

    # gewünschte Signal d
    D = Ys +N

    # Systemstoßantwort ws 
    ntaps = 50  # länge der Systemstoßantwort
    norm = 1/normYs

    dirac = np.zeros(ntaps)  # dirac stoß
    dirac[0] = 1

    ws = sg.lfilter(b, a, dirac) * norm
    
    return X, D, ws, norm

def wiener_loesung(N, N_stat, sigma2Xnoise, sigma2Xcos, alphaXcos, ws, sn2):
    """

    @param N: Anz. Filterkoeffizienten = Filterordnung+1
    @param N_stat: Faltungslänge
    @param sigma2Xnoise: Varianz  weisses Rauschen
    @param sigma2Xcos: Varianz cos
    @param alphaXcos: Normierte Frequenz cos
    @param ws: ystemstoßantwort (gegeben durch b und a)
    @param sn2:
    @return: p - Kreuzkorrelationsvektor, R - Autokorr. Matrix, rx - Autokorr. Funktion, ry - Autokorr. Funktion von d', sd2 - Varianz von d
    """
    R, rx = calc_R(sigma2Xnoise, sigma2Xcos, alphaXcos, N, N_stat)
    ryx = calc_ryx(ws, 1, rx, N_stat)
    p = ryx[0:N]

    ry = calc_ry(ws, 1, rx, N_stat)
    sd2 = ry[0] + sn2
 
    # Wiener Lösung
    w0 = np.linalg.inv(R) @ p

    return p, R, ry, sd2, w0




