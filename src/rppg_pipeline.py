import numpy as np
from scipy.signal import butter, filtfilt, spectrogram, welch

def butter_bandpass(signal, lowcut=0.75, highcut=2.5, fs=30.0, order=3):
    nyq = 0.5 * fs
    low = max(0.01, min(lowcut / nyq, 0.98))
    high = max(low + 0.01, min(highcut / nyq, 0.99))
    
    b, a = butter(order, [low, high], btype='band')
    default_padlen = 3 * max(len(a), len(b))
    safe_padlen = min(default_padlen, len(signal) - 1)
    
    if safe_padlen <= 0:
        return signal
    return filtfilt(b, a, signal, padlen=safe_padlen)

def extract_bvp_green(rgb_series, fps=30.0):
    """
    GREEN rPPG extraction with edge-padded detrending to prevent boundary transients.
    """
    if rgb_series is None or len(rgb_series) < 4 or rgb_series.ndim < 2 or rgb_series.shape[1] < 2:
        return np.array([])

    g_channel = rgb_series[:, 1].astype(np.float64)
    
    window = max(3, int(round(fps * 1.2)))
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded_g = np.pad(g_channel, (pad_left, pad_right), mode='edge')
    smooth_g = np.convolve(padded_g, np.ones(window, dtype=np.float64) / window, mode='valid')
    detrended_g = g_channel - smooth_g
    
    bvp = butter_bandpass(detrended_g, lowcut=0.75, highcut=2.5, fs=fps)
    return (bvp - np.mean(bvp)) / (np.std(bvp) + 1e-6)

def extract_bvp_chrom(rgb_series, fps=30.0):
    """
    CHROM rPPG extraction: linear combination of RGB channels to minimize
    lighting drift, motion noise, and MP4 compression distortion.
    """
    if rgb_series is None or len(rgb_series) < 4 or rgb_series.ndim < 2 or rgb_series.shape[1] < 3:
        return np.array([])

    rgb = rgb_series.astype(np.float64)
    window = max(3, int(round(fps * 1.2)))
    pad_l, pad_r = window // 2, window - 1 - (window // 2)
    kernel = np.ones(window, dtype=np.float64) / window
    
    norm_rgb = np.zeros_like(rgb)
    for c in range(3):
        padded = np.pad(rgb[:, c], (pad_l, pad_r), mode='edge')
        local_mean = np.convolve(padded, kernel, mode='valid')
        norm_rgb[:, c] = rgb[:, c] / (local_mean + 1e-6)
        
    xs = 3.0 * norm_rgb[:, 0] - 2.0 * norm_rgb[:, 1]
    ys = 1.5 * norm_rgb[:, 0] + norm_rgb[:, 1] - 1.5 * norm_rgb[:, 2]
    
    xf = butter_bandpass(xs, lowcut=0.75, highcut=2.5, fs=fps)
    yf = butter_bandpass(ys, lowcut=0.75, highcut=2.5, fs=fps)
    
    alpha = np.std(xf) / (np.std(yf) + 1e-6)
    bvp = xf - alpha * yf
    return (bvp - np.mean(bvp)) / (np.std(bvp) + 1e-6)

def extract_bvp_hybrid(rgb_series, fps=30.0):
    """
    Blends GREEN (sensitive to synthetic RGB blending artifacts) with CHROM
    (robust against lighting/compression noise on real-world MP4s).
    """
    bvp_g = extract_bvp_green(rgb_series, fps=fps)
    bvp_c = extract_bvp_chrom(rgb_series, fps=fps)
    if len(bvp_g) == 0:
        return np.array([])
    if len(bvp_c) == 0:
        return bvp_g
    combined = 0.65 * bvp_g + 0.35 * bvp_c
    return (combined - np.mean(combined)) / (np.std(combined) + 1e-6)

def _parabolic_peak_freq(freqs, psd_slice):
    """
    Sub-bin parabolic interpolation around the spectral peak to prevent
    FFT bin quantization jumps on real videos.
    """
    idx = int(np.argmax(psd_slice))
    if 0 < idx < len(freqs) - 1:
        y0, y1, y2 = np.log(psd_slice[idx - 1] + 1e-12), np.log(psd_slice[idx] + 1e-12), np.log(psd_slice[idx + 1] + 1e-12)
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            df = freqs[1] - freqs[0]
            return float(freqs[idx] + np.clip(delta, -0.5, 0.5) * df)
    return float(freqs[idx])

def compute_hr_welch(bvp_signal, fs=30.0):
    """
    Computes dominant stationary heart rate (in BPM) via Welch PSD.
    """
    if len(bvp_signal) < int(fs * 2):
        return 75.0
    nperseg = min(len(bvp_signal), int(fs * 5.0))
    freqs, psd = welch(bvp_signal, fs=fs, nperseg=nperseg, nfft=2048)
    mask = (freqs >= 0.75) & (freqs <= 2.5)
    if not np.any(mask):
        return 75.0
    peak_hz = _parabolic_peak_freq(freqs[mask], psd[mask])
    return peak_hz * 60.0

def compute_hr_stft(bvp_signal, fs=30.0, window_sec=4.0, step_sec=0.1):
    """
    Short-Time Fourier Transform (STFT) using the paper's optimal 0.1s extraction interval.
    """
    nperseg = max(4, int(round(window_sec * fs)))
    step_samples = max(1, int(round(step_sec * fs)))
    noverlap = max(1, min(nperseg - 1, nperseg - step_samples))
    
    if len(bvp_signal) < nperseg:
        return np.array([])
        
    frequencies, _, Sxx = spectrogram(
        bvp_signal, 
        fs=fs, 
        nperseg=nperseg, 
        noverlap=noverlap, 
        nfft=2048,
        scaling='density'
    )
    
    valid_mask = (frequencies >= 0.75) & (frequencies <= 2.5)
    if not np.any(valid_mask):
        return np.array([])
        
    valid_freqs = frequencies[valid_mask]
    valid_Sxx = Sxx[valid_mask, :]
    
    hr_estimates = [
        _parabolic_peak_freq(valid_freqs, valid_Sxx[:, t_idx]) * 60.0
        for t_idx in range(valid_Sxx.shape[1])
    ]
    return np.array(hr_estimates, dtype=np.float64)