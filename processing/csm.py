import numpy as np

class CSM():
    def __init__(self, signal, sample_freq, block_size, calib=None, num_blocks=1, precision=np.float32):
        self.signal = signal
        self.sample_freq = sample_freq
        self.block_size = block_size
        self.window_ = np.hanning
        self.calib = calib
        self.num_blocks = num_blocks
        self.precision = precision

    def get_source_data(self):
        num_samples = self.signal.shape[0]
        num_channels = self.signal.shape[1]

        for start in range(0, num_samples, self.block_size):
            end = start + self.block_size
            if end > num_samples:
                break 
            yield self.signal[start:end, :]

    def calc_csm(self):
        """Berechnung der Cross-Spectral Matrix (CSM)."""
        t = self.signal
        wind = self.window_(self.block_size)
        weight = np.dot(wind, wind)
        wind = wind[np.newaxis, :].swapaxes(0, 1)
        numfreq = int(self.block_size / 2 + 1)
        csm_shape = (numfreq, t.shape[1], t.shape[1])
        csm_upper = np.zeros(csm_shape, dtype=self.precision)

        if self.calib and self.calib.num_mics > 0:
            if self.calib.num_mics == t.shape[1]:
                wind = wind * self.calib.data[np.newaxis, :]
            else:
                raise ValueError('Calibration data not compatible: %i, %i' % (self.calib.num_mics, t.shape[1]))

        for data in self.get_source_data():
            ft = np.fft.rfft(data * wind, axis=0).astype(self.precision)
            for freq_idx in range(numfreq):
                csm_upper[freq_idx] += np.outer(ft[freq_idx], np.conj(ft[freq_idx]))
        
        csm_lower = csm_upper.conj().transpose(0, 2, 1)
        [np.fill_diagonal(csm_lower[cntFreq, :, :], 0) for cntFreq in range(csm_lower.shape[0])]
        csm = csm_lower + csm_upper
        
        return csm * (2.0 / self.block_size / weight / self.num_blocks)
    
    
def calculate_octave_band_csm(csm_result, freqs, center_freqs):
    band_csm = []
    factor = 2**(1/6) 
    
    for center_freq in center_freqs:
        lower_freq = center_freq / factor
        upper_freq = center_freq * factor
        freq_indices = np.where((freqs >= lower_freq) & (freqs <= upper_freq))[0]
        band_csm_mean = np.mean(csm_result[freq_indices], axis=0)
        band_csm.append(band_csm_mean)

    return np.array(band_csm)

def calculate_combined_csm(csm_result, freqs, center_freqs):
    combined_csm = np.zeros((csm_result.shape[1], csm_result.shape[2]), dtype=csm_result.dtype)
    factor = 2**(1/6) 
    
    for center_freq in center_freqs:
        lower_freq = center_freq / factor
        upper_freq = center_freq * factor
        freq_indices = np.where((freqs >= lower_freq) & (freqs <= upper_freq))[0]
        combined_csm += np.sum(csm_result[freq_indices], axis=0)
    
    return combined_csm