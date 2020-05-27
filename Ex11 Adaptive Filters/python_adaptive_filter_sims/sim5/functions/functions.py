import math
import numpy as np
import time

from scipy import signal as sg


def calc_R2(b, a, N, N_stat):
    """

    :param b: Filterkoeffizient
    :param a: Filterkoeffizient
    :param N: Anz. Filterkoeffizienten = Filterordnung+1
    :param N_stat: Faltungslänge, je länger rx oder die Systemstoßantwort,
        desto größer muss N_stat sein, um eine Gleiche Genauigkeit zu erreichen
    :return: ry - Autokorrelationsfunktion von y
    """
    gamma = np.zeros(N_stat)
    gamma[0] = 1  # gamma Impuls
    ry = calc_ry_freq(b, a, gamma, N_stat)

    return ry[0:N_stat] / ry[0]


def calc_ry_freq(b, a, rx, N_stat):
    """
    Berechnung der Autokorrelationsfunktion ry des Ausgangs eines Filters.
    Berechnung der Faltung im Frequenzbereich
    ry_freq = H^2 * rx_freq

    :param b: Filterkoeffizient
    :param a: Filterkoeffizient
    :param rx: Autokorrelationsfunktion von x
    :param N_stat: Faltungslänge, je länger rx oder die Systemstoßantwort,
        desto größer muss N_stat sein, um eine Gleiche Genauigkeit zu erreichen
    :return: ry - Autokorrelationsfunktion von y
    """
    N_stat = 2 ** (math.ceil(np.log2(N_stat)))

    # Autokorrelationsfunktion von x
    rx1 = np.concatenate([rx, np.zeros(N_stat - len(rx))])  # einseitig
    rx2 = np.concatenate(
        [np.zeros(2 * N_stat), np.flip(rx1[2 : len(rx1)]), rx1]
    )  # zweiseitige Ergänzung und Verschiebung nach Rechts -> Lineare Faltung

    # Systemübertragungsfunktion
    H = np.fft.fft(b, n=4 * N_stat) / np.fft.fft(a, n=4 * N_stat)

    # Leistungsdichtespektrum von y, ry_freq = H^2 * Rx
    ry_freq = np.abs(H) ** 2 * np.fft.fft(rx2, n=4 * N_stat)

    # Autokorrelationsfunktion von ry
    ry = np.real(np.fft.ifft(ry_freq))

    return ry[3 * N_stat : 4 * N_stat]


def calc_ryx_freq(b, a, rx, N_stat):
    """
    Berechnung der Kreuzkorrelationsfunktion ryx
    Wie calc_ryx.m, jedoch wird die Faltung im Frequenzbereich berechnet.

    x: Eingang , y: Ausgang des Filters mit den Koeffizienten b, a
    h: Systemstossantwort

    Berechnung: ryx[.] = h[.] * rx

    :param b: Filterkoeffizient
    :param a: Filterkoeffizient
    :param rx: Autokorrelationsfunktion
    :param N_stat:  Faltungslänge. Je länger rx oder die Systemstoßantwort,
        desto größer muss N_stat sein, um eine gleiche Genauigkeit zu erreichen.
    :return: ryx - Kreuzkorrelationsfunktion
    """
    N_stat = 2 ** math.ceil(math.log2(N_stat))

    # Autokorrelationsfunktion von x
    rx1 = np.concatenate([rx, np.zeros(N_stat - len(rx))])  # einseitig
    rx2 = np.concatenate(
        [np.zeros(2 * N_stat + 1), np.flip(rx1[2 : len(rx1)])]
    )  # zweiseitige Ergänzung und Verschiebung
    # nach rechts -> lineare Faltung

    # Systemübertragungsfunktion Frequenzbereich
    H = np.fft.fft(b, 4 * N_stat) / np.fft.fft(a, 4 * N_stat)

    # Leistungsdichtespektrum ryx_freq = H Rx
    ryx_freq = H * np.fft.fft(rx2, 4 * N_stat)

    # Autokorrelationsfunktion von y
    ryx_time = np.real(np.fft.ifft(ryx_freq))
    ryx_time = ryx_time[3 * N_stat + 1 : 4 * N_stat]  # Nullpunkt bei ry[0]
    return ryx_time


def expwin2(func_in, factor=0.1):
    """
    Glättung (zeitliche Mittelung) der Eingangssequenz  func_in
    mit Hilfe eines exponentiellen Fensters.
    Start bei mean(1:lge(func_in)/100)

    :param func_in: Eingangssequenz
    :param factor: Vergessensfaktor (0 < factor < 1) je näher bei 1,
        desto größer die Gewichtung des neuen Wertes, d.h desto schwächer die
        Glättung.
    :return: Geglättetes Signal
    """
    func_out = np.zeros(np.size(func_in))
    func_out[0] = np.mean(func_in[0 : math.ceil(len(func_in) / 100)])

    for idx in range(1, len(func_in)):
        func_out[idx] = factor * func_in[idx] + (1 - factor) * func_out[idx - 1]

    return func_out


def signal_ar(len, sigma2N, b, a, ar_flag, ar_koeff):
    """
    Erzeugung der Signale x,d
    Eingangssignal: weisses Rauschen oder  AR-Prozess
    AR-Prozess       falls ar_flag = True
    Erwüenschtes Signal d = Systemausgang + Messrauschen

    :param len: Signallänge
    :param sigma2N: Varianz des weißen Messrauschens
    :param b: Filterkoeffizienten des Systemfilters
    :param a: Filterkoeffizienten des Systemfilters
    :param ar_flag: False: weißes Rauschen, True: AR-Prozess
    :param ar_koeff: LPC Koeffizienten zur Erzeugung eines AR-Prozesses mit
        sprachtypischem Spektrum
    :return: x - Eingangssignal  , d - Erwünschte Signal,
        ws - Systemstoßantwort, norm - Normierung des Systemfilters
    """
    x = np.random.randn(len)  # weißes Rauschen

    if ar_flag:
        x = sg.lfilter(np.array([1]), ar_koeff, x)  # AR Prozess

    normX = np.sqrt(np.var(x))  # Normierung der Eingangsleisgtung auf var(x)=1

    x = x / normX

    # Systemausgang Ys
    Ys = sg.lfilter(b, a, x)  # Systemausgang Ys
    normYs = np.sqrt(np.var(Ys))  # Normierung des Systenausgangs auf var(Ys)=1
    Ys = Ys / normYs

    # weißes Messrauschen
    N = np.sqrt(sigma2N) * np.random.randn(len)

    # erwünschtes Signal D
    d = Ys + N  # Systemausgang + Messrauschen

    norm = 1 / normYs
    ws = b * norm
    return x, d, ws, norm


def flms(mu, C, L, x, d, gamma=0.6):
    """
    FLMS-Algorithmus

    :param mu: Schrittweite
    :param N: Anz. Filterkoeffizienten = Filterordnung+1
    :param C: DFT Länge
    :param L: Blockverschiebung
    :param x: Filtereingang x[.]
    :param d: erwünschtes Signal d[.]
    :param w_start: Startwerte Filterkoeffizienten
    :param gamma: Vergessensfaktor
    :return: e_time - Fehlervektor (Zeitbereich),
        ws_time - Gewichtsvektor (Zeitbereich), d_hat - Filterausgang,
        Schätzung von d, calc - Benötigte Zeit zum berechnen,
        p_hat - Schätzung des Leistungsdichtespektrum
    """
    # Initialisierungen
    e_time = np.zeros(len(x))  # Fehlervektor
    d_hat = np.zeros(len(x))  # Filterausgang, Schätzung von d
    ws_freq = np.zeros(C)  # Gewichtsvektor im Frequenzbereich
    p_hat = C * np.ones(C)  # Schätzung des Leistungsdichtespektrum

    # FLMS Parameter
    numblk = math.floor(len(x) / L)  # Anzahl Verarbeitungsblöcke
    begin = math.ceil(C / L) - 1  # erster Blockindex

    start_time = time.time()
    for idx in range(begin, numblk):
        # DFT von c, Länge C
        x_freq = np.fft.fft(
            x[(idx + 1) * L - C : (idx + 1) * L]
        )  # aktuellster Wert des Blockes (idx+1)*L

        # Filterung im Frequenzbereich
        y_freq = x_freq * ws_freq
        y_time = np.real(np.fft.ifft(y_freq))  # IDFT, Filterausgang im Zeitbereich

        # Fehlersignal im Zeitbereich
        # e = d - y
        e_time[idx*L : (idx + 1)*L] = d[idx*L : (idx + 1)*L] - y_time[C-L : C]
        # Overlap-Save-Verfahren: nur die letzten L Werte von y_time werden
        # verwendet, um ein lineare Faltung zu erhalten

        # Fehlersifnal im Frequenzbereich
        e_freq = np.fft.fft(
            np.concatenate((np.zeros(C - L), e_time[idx * L : (idx + 1) * L]))
        )

        # update von P_hat (Schätzwert)
        p_hat = np.abs((1 - gamma) * np.conj(x_freq) * x_freq + gamma * p_hat)

        # update von ws_freq
        ws_freq = (
            ws_freq + mu * np.conj(x_freq) / (p_hat + 0.001) * e_freq
        )  # Punktweise Division, Adaption von ws_freq

        # Projektion (letzten L-1 Werte von w zu Null setzen)
        ws_time = np.real(np.fft.ifft(ws_freq))
        ws_freq = np.fft.fft(ws_time[0 : C - L], C)

        d_hat[idx * L : (idx + 1) * L] = y_time[C - L : C]

    end_time = time.time()
    calc = end_time - start_time
    ws_time = np.concatenate(
        [ws_time[0 : C - L], np.zeros(L - 1)]
    )  # Filterkoeffizienten am Ender der Adaption

    return e_time, ws_time, d_hat, calc, p_hat


def lms(mu, N, x, d, w_start, adaptstart):
    """
    LMS-Algorithmus,
    wie lms.m, jedoch:
        -  Adaptionstart bei: begin
        -  unterbrechbar
        -  Rechenzeit wird ausgegeben
    :param mu: Schrittweite
    :param N: Anz. Filterkoeffizienten = Filterordnung+1
    :param x: Filtereingang X[.]
    :param d: erwünschtes Signal D[.]
    :param w_start: Startwerte Filterkoeffizienten
    :param adaptstart: Adaptionstartpunkt
    :return: e_mat - Fehlersignal, w - Gewichtsvektor nach Adaptionsende,
        d_hat - Filterausgang, Schätzung von d, calc - Berechnungszeit
    """
    # Initialisierungen
    adaptlen = len(x)
    w = w_start

    d_hat = np.zeros(adaptlen)
    e_mat = np.zeros(adaptlen)

    start_time = time.time()

    for idx in range(adaptstart, adaptlen):
        x_ = x[idx - N : idx]
        x_ = np.flip(x_)

        y = np.transpose(x_) @ w
        e = d[idx] - y  # erwünschtes Signal - Filterausgang
        w = w + mu * e * x_  # berechnung der neuen Koeffizienten

        e_mat[idx] = e
        d_hat[idx] = y

    stop_time = time.time()

    calc = stop_time - start_time

    return e_mat, w, d_hat, calc
