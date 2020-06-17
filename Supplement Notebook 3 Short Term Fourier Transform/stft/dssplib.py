"""Function library for digital speech signal processing"""
import numpy as np
import scipy.signal as sig
import scipy.fftpack as sp_fft
import matplotlib.pyplot as plt
from IPython.display import display, Audio


def stft(x, fft_length=1024, frame_shift=128,
         window_length=512, window='boxcar'):
    w = sig.get_window(window, window_length)

    # zero padding so that (total_samples - window_length)
    # is a multiple of the frame shift
    num_zeros = int(np.ceil((len(x)- window_length)/frame_shift)*frame_shift
                    + window_length - len(x))
    x = np.concatenate((x, np.zeros(num_zeros)))

    # preallocate the result matrix
    num_frames = (len(x) - window_length)//frame_shift
    X = np.zeros((fft_length, num_frames), dtype=np.complex128)

    # processing
    for m in range(num_frames):
        X[:, m] = np.fft.fft(x[m*frame_shift:m*frame_shift+window_length]*w,
                             fft_length)
    return X


def istft(X, window_length=512, frame_shift=128,
          fft_length=1024, window='boxcar'):
    nbins, nframes = X.shape
    if nbins == fft_length//2 +1:
        ifft = np.fft.irfft
    elif nbins == fft_length:
        ifft = np.fft.ifft
    else:
        raise ValueError('fft dimension mismatch')

    x_out = np.zeros((nframes - 1)*frame_shift + fft_length)

    w = sig.get_window(window, window_length)

    for m in range(nframes):
        X_transform = X[:fft_length//2 +1, m].copy()

        x_n = np.fft.irfft(X_transform, axis=0)
        x_out[m*frame_shift:m*frame_shift + window_length] \
                += w* x_n[:window_length].real

    return x_out


def mel_filterbank(nfilters=9, fft_length=1024, sample_rate=16000,
                   lowfreq=0, highfreq=None):

    def hz2mel(f):
        return 2595*np.log(1 + f/700)

    def mel2hz(mel):
        return 700*(np.exp(mel/2595) - 1)

    highfreq = highfreq or sample_rate/2
    assert highfreq <= sample_rate/2
    assert lowfreq <= highfreq

    mel_center = np.linspace(hz2mel(lowfreq), hz2mel(highfreq), nfilters+2)
    bin_center = np.round(mel2hz(mel_center)*fft_length/sample_rate)
    bin_center = np.unique(bin_center).astype(int)
    nfilters = len(bin_center) - 2

    filter_banks = np.zeros((nfilters, fft_length//2 + 1), dtype=np.float64)

    for j in range(nfilters):
        for k in range(bin_center[j], bin_center[j+1]):
            filter_banks[j, k] = ((k - bin_center[j])
                                  / (bin_center[j+1]-bin_center[j]))
        for k in range(bin_center[j+1], bin_center[j+2]):
            filter_banks[j, k] = ((bin_center[j+2] - k)
                                  / (bin_center[j+2]-bin_center[j+1]))
    return filter_banks, nfilters, bin_center*sample_rate/fft_length


def mfcc(x, sample_rate=16000,
         nfilters=25, ncep=13,
         frame_shift=128, fft_length=1024,
         window_length=512, window='boxcar'):
    # STFT
    X = stft(x, frame_shift=frame_shift, fft_length=fft_length,
             window_length=window_length, window=window)
    # periodogram
    X_power = np.abs(X)**2/fft_length

    # Mel filterbank binning
    filter_banks, nfilters, f_center = mel_filterbank(
        fft_length=fft_length, sample_rate=sample_rate, nfilters=nfilters)
    X_mel = filter_banks @ X_power[:fft_length//2+1]

    # To log-mel spectrum
    X_log_mel = np.log(X_mel)

    # to cepstrum
    X_cep = sp_fft.dct(X_log_mel, n=nfilters,
                       type=2, norm='ortho', axis=0)

    # extract phase (only for later reconstruction!)
    X_phase = np.exp(1j*np.angle(X))
    # drop upper coefficients
    return X_cep[:ncep], X_phase, f_center, filter_banks


def inv_mfcc(X_mfcc, X_phase, sample_rate=16000,
             nfilters=25, filter_banks=None,
             frame_shift=128, fft_length=1024,
             window_length=512, window='boxcar'):
    ncep, nframes = X_mfcc.shape

    nfilters = filter_banks.shape[0] or nfilters

    if filter_banks is None:
        filter_banks, nfilters, _ = mel_filterbank(
            fft_length=fft_length, sample_rate=sample_rate,
            nfilters=nfilters)

    # undo discrete cosine transform
    X_log_mel = sp_fft.idct(X_mfcc, n=nfilters,
                            type=2, norm='ortho', axis=0)

    # inverse mel filter bank binning
    X_mel = np.exp(X_log_mel)
    X_power = np.abs(np.linalg.pinv(filter_banks) @ X_mel)

    # reconstruct STFT with periodogram and phase
    X_stft_abs = np.sqrt(X_power)*fft_length
    X_stft = X_stft_abs * X_phase[:fft_length//2 +1]

    x_out = istft(X_stft, frame_shift=frame_shift, fft_length=fft_length,
                  window_length=window_length, window=window)
    return x_out, X_stft


def plot_time_domain_signal(x, sample_rate, title="", ylabel=r'$x(t)$'):
    display(Audio(x, rate=sample_rate))
    time = np.arange(0, len(x))*1/sample_rate
    plt.plot(time, x)

    plt.title(title, fontsize=18)
    plt.xlabel(r'$t / \mathrm {s}$', fontsize=15)
    plt.xlim((0, time.max()))
    xlocs = np.arange(0, time.max(), 0.25)
    plt.xticks(xlocs, ['{:.2f}'.format(xloc) for xloc in xlocs],
               fontsize=15)

    plt.ylabel(ylabel, fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)


def plot_stft(X_stft, frame_shift=128, sample_rate=16000,
              floor_val=-70, cut_upper=True):

    fft_length, num_frames = X_stft.shape

    # compute periodogram
    # only use lower half because of symmetry property of DFT: X(N-k) = X(k)
    if cut_upper:
        X_abs = np.abs(X_stft[:fft_length//4+1, :])
    else:
        X_abs = np.abs(X_stft)

    if floor_val is not None:
        # floor values
        X_abs_min = 10**(floor_val/20)*X_abs.max()
        X_abs = np.maximum(X_abs, X_abs_min)
        X_plot = 20*np.log(X_abs)

    # plot as image: Transpose to get time as x-axis and flip upside down.
    # Show only the lower half because the spectrum of a real-valued signal
    # is symmetric around the middle frequency
    plt.imshow(X_plot, origin='lower', aspect='auto')

    # label time axis: time in seconds
    times = np.arange(0, num_frames*frame_shift/sample_rate, 0.25)
    xlocs = np.round((times*sample_rate)/frame_shift).astype(int)
    plt.xticks(xlocs, ["{:.2f}".format(time) for time in times],
               fontsize=15)
    plt.xlabel(r'$t / \mathrm{s}$', fontsize=15)

    # label frequency axis: frequency in kHz
    ylocs = np.round(np.linspace(0, X_plot.shape[0], 5)).astype(int)
    plt.yticks(ylocs,
               ["{:.0f}".format(sample_rate*l/fft_length/1000) for l in ylocs],
               fontsize=15)
    plt.ylabel(r'$f /\mathrm{kHz}$', fontsize=15)


def plot_mel_spectogram(X_mel, freqs=None, sample_rate=16000,
                        frame_shift=128, floor_val=-70, scale='log'):
    nfilters, num_frames = X_mel.shape
    assert nfilters == len(freqs)-2, f'{nfilters} != {len(freqs)-2}'

    if floor_val is not None:
        if scale != 'log':
            X_mel = np.exp(X_mel)
        X_mel = np.abs(X_mel)
        X_mel_min = 10**(floor_val/20)*X_mel.max()
        X_mel = np.maximum(X_mel, X_mel_min)
        X_plot = np.log(X_mel)

    plt.imshow(X_plot, origin='lower', aspect='auto')
    if freqs is not None:
        plt.yticks(np.arange(nfilters)[::len(freqs)//6+1],
                   ['{:.2f}'.format(f/1000) for f in freqs[::len(freqs)//6+1]],
                   fontsize=15)
        plt.ylabel(r'$f / \mathrm{kHz}$', fontsize=15)

    # label time axis: time in seconds
    if frame_shift is not None and sample_rate is not None:
        times = np.arange(0, num_frames*frame_shift/sample_rate, 0.25)
        xlocs = np.round((times*sample_rate)/frame_shift).astype(int)
        plt.xticks(xlocs,
                   ["{:.2f}".format(time) for time in times],
                   fontsize=15)
        plt.xlabel(r'$t / \mathrm{s}$', fontsize=15)


def plot_time_domain_and_spectrogram(x, sample_rate=16000,
                                     ylabel='', floor_val=-45):
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plot_time_domain_signal(x, sample_rate=sample_rate, ylabel=ylabel)
    X_stft = stft(x, window='hamming')
    plt.subplot(2, 1, 2)
    plot_stft(X_stft, sample_rate=sample_rate, floor_val=floor_val)
    plt.show()
