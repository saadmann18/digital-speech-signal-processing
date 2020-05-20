import numpy as np
from matplotlib import pyplot as plt

def mel_filterbank(sampling_rate, num_channels, fft_length):
    # -------------------------------------------------------------------------
    # Triangular mel - windows (normalized to sum to 1)
    #
    #     / \           L = LowIndex, H = HighIndex
    # y1 /   \ y2
    #   /     \         y1(k) = (x-L+1)/(k-L+1)
    #  /       \        y2(k) = (H+1-x)/(H+1-k)
    #  L   k    H    ==> y(k)=(y1(k),y2(k)),  mel_bank = (mel_bank; y(k))

    def hz2mel(freq_hz):
        return 2595*np.log10( 1 + freq_hz/ 700)

    def mel2hz(freq_mel):
        return 700*(10**(freq_mel / 2595) - 1)

    nyquist_bin = fft_length // 2 + 1
    nyquist_freq_mel = hz2mel(sampling_rate / 2)
    half_channel_size = nyquist_freq_mel / (2 * num_channels)
    delta_hz = sampling_rate / (2 * fft_length)
    
    low_index = np.empty(nyquist_bin, dtype=np.int32)
    high_index = np.empty(nyquist_bin, dtype=np.int32)
    for k in range(nyquist_bin):
        freq_mel = hz2mel(k * delta_hz)

        low_freq_hz = max(mel2hz(freq_mel - half_channel_size), 0)
        low_index[k] = int(round(low_freq_hz / delta_hz) + 1)

        high_freq_hz = min(mel2hz(freq_mel + half_channel_size),
                           sampling_rate/2)
        high_index[k] = min(int(round(high_freq_hz / delta_hz) + 1),
                            nyquist_bin - 1)

    mel_range = [(low, high) for low, high in zip(low_index, high_index)]
    raw_size = np.max(high_index - low_index) + 1
    mel_bank = np.zeros((nyquist_bin, raw_size))

    for k in range(nyquist_bin):
        i = 0
        for x in range(low_index[k], k):
            mel_bank[k, i] = (x - low_index[k] + 1) / (k - low_index[k] + 1)
            i += 1

        for x in range(k+1, high_index[k]):
            mel_bank[k, i] = (high_index[k] + 1 - x) / (high_index[k] + 1 - k)
            i += 1
    # normalize along mel axis so that weights sum to 1
    mel_bank /= mel_bank.sum(axis=1)[:, np.newaxis]

    return mel_range, mel_bank


if __name__ == '__main__':
    idxs, mel_bank = mel_filterbank(16000, 10, 512)
    plt.plot(mel_bank.T)
    plt.show()
