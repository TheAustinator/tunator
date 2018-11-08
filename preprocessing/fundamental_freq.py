from collections import deque
import itertools
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt

import ipdb


RING_BUFFER_SIZE = 40
SAMPLE_RATE = 22050
# THRESHOLD_MULTIPLIER = 5
WINDOW_SIZE = 2048
# THRESHOLD_WINDOW_SIZE = 7

np.set_printoptions(threshold=np.nan)


class SpectralAnalyzer:

    FREQUENCY_RANGE = (100, 1000)

    def __init__(self, window_size, segments_buf=None):
        self._window_size = window_size
        if segments_buf is None:
            segments_buf = int(SAMPLE_RATE / window_size)
        self._segments_buf = segments_buf

        self._thresholding_window_size = THRESHOLD_WINDOW_SIZE
        assert self._thresholding_window_size <= segments_buf

        self._last_spectrum = np.zeros(window_size, dtype=np.int16)
        self._last_flux = deque(
            np.zeros(segments_buf, dtype=np.int16), segments_buf)
        self._last_prunned_flux = 0

        self._hanning_window = np.hanning(window_size)
        # The zeros which will be used to double each segment size
        self._inner_pad = np.zeros(window_size)

        # To ignore the first peak just after starting the application
        self._first_peak = True

    def _get_flux_for_thresholding(self):
        return list(itertools.islice(
            self._last_flux,
            self._segments_buf - self._thresholding_window_size,
            self._segments_buf))

    def find_onset(self, spectrum):
        """
        Calculates the difference between the current and last spectrum,
        then applies a thresholding function and checks if a peak occurred.
        """
        last_spectrum = self._last_spectrum
        flux = sum([max(spectrum[i] - last_spectrum[i], 0)
            for i in range(self._window_size)])
        self._last_flux.append(flux)

        thresholded = np.mean(
            self._get_flux_for_thresholding()) * THRESHOLD_MULTIPLIER
        prunned = flux - thresholded if thresholded <= flux else 0
        peak = prunned if prunned > self._last_prunned_flux else 0
        self._last_prunned_flux  = prunned
        return peak

    def find_fundamental_freq(self, samples):
        cepstrum = self.cepstrum(samples)
        # search for maximum between 0.08ms (=1200Hz) and 2ms (=500Hz)
        # as it's about the recorder's frequency range of one octave
        min_freq, max_freq = self.FREQUENCY_RANGE
        start = int(SAMPLE_RATE / max_freq)
        end = int(SAMPLE_RATE / min_freq)
        narrowed_cepstrum = cepstrum[start:end]

        peak_ix = narrowed_cepstrum.argmax()
        freq0 = SAMPLE_RATE / (start + peak_ix)

        if freq0 < min_freq or freq0 > max_freq:
            # Ignore the note out of the desired frequency range
            return

        return freq0

    def process_data(self, data):
        spectrum = self.autopower_spectrum(data)

        onset = self.find_onset(spectrum)
        self._last_spectrum = spectrum

        if self._first_peak:
            self._first_peak = False
            return

        if onset:
            freq0 = self.find_fundamental_freq(data)
            return freq0

    def autopower_spectrum(self, samples):
        """
        Calculates a power spectrum of the given data using the Hamming window.
        """
        # TODO: check the length of given samples; treat differently if not
        # equal to the window size

        windowed = samples * self._hanning_window
        # Add 0s to double the length of the data
        padded = np.append(windowed, self._inner_pad)
        # Take the Fourier Transform and scale by the number of samples
        spectrum = np.fft.fft(padded) / self._window_size
        autopower = np.abs(spectrum * np.conj(spectrum))
        return autopower[:self._window_size]

    def cepstrum(self, samples):
        """
        Calculates the complex cepstrum of a real sequence.
        """
        spectrum = np.fft.fft(samples)
        log_spectrum = np.log(np.abs(spectrum))
        cepstrum = np.fft.ifft(log_spectrum).real
        return cepstrum


os.chdir('working_dir')
song, sample_rate = librosa.load('vocals_isolated_phased.wav')

# set up iteration to vary THRESHOLD_MULTIPLIER and THRESHOLD_WINDOW_SIZE
start_1 = 2
stop_1 = 11
step_1 = 2
range_1 = range(start_1, stop_1, step_1)
dim_1 = len(range_1)
start_2 = 6
stop_2 = 20
step_2 = 2
range_2 = range(start_2, stop_2, step_2)
dim_2 = len(range_2)
fig = plt.figure()

# Jake VanDerplas implementation
# for i in range(2):
#     for j in range(3):
#         ax[i, j].text(0.5, 0.5, str((i, j)),
#                       fontsize=18, ha='center')

# vary THRESHOLD_MULTIPLIER
for i in range_1:
    THRESHOLD_MULTIPLIER = i

    # vary THRESHOLD_WINDOW_SIZE
    for j in range_2:
        THRESHOLD_WINDOW_SIZE = j

        fig
        # set up plot
        col = (j - start_2) / step_2
        row = (i - start_1) / step_1
        plt_i = (row * dim_2) + col + 1
        print(f'i: {i}\nj: {j}\ncol: {col}\nrow: {row}\nplt_i: {plt_i}')
        ax = fig.add_subplot(dim_1, dim_2, plt_i)

        # initialize spectral analyzer with new globals
        spectral_analyzer = SpectralAnalyzer(
            window_size=WINDOW_SIZE,
            segments_buf=RING_BUFFER_SIZE,
        )

        freq_arr = np.array([])
        # iterate over wav file in chunks of WINDOW_SIZE
        for k in range(round(len(song) / WINDOW_SIZE)):
            window_start = k * WINDOW_SIZE
            window_end = (k + 1) * WINDOW_SIZE
            data_array = song[window_start:window_end]
            freq = spectral_analyzer.process_data(data_array)
            freq_arr = np.append(freq_arr, [freq])
        ax.scatter(range(len(freq_arr)), freq_arr, s=4)
        ax.title.set_text(f'MULTIPLIER: {i}, WINDOW: {j}')

plt.subplots_adjust(hspace=.5)
plt.show()

