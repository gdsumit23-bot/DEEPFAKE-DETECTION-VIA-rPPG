import numpy as np
from scipy.signal import welch

# ────────────────────────────────────────────────────────────────────────────
# Feature 1 — Temporal HRV (SDNN)
# ────────────────────────────────────────────────────────────────────────────
def compute_feat1_sdnn(hr_values):
    """Standard deviation of HR series — measures temporal stability."""
    if hr_values is None or len(hr_values) == 0:
        return 0.0
    return float(np.std(hr_values))

def compute_feat2_rmssd(hr_values):
    """Root mean square of successive HR differences — short-term variability."""
    if hr_values is None or len(hr_values) < 2:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(hr_values) ** 2)))

def compute_feat3_sdsd(hr_values):
    """Standard deviation of successive differences."""
    if hr_values is None or len(hr_values) < 2:
        return 0.0
    diffs = np.abs(np.diff(hr_values))
    return float(np.std(diffs))


# ────────────────────────────────────────────────────────────────────────────
# Feature 4 — Pulse Autocorrelation (PAC)
# ────────────────────────────────────────────────────────────────────────────
def compute_pulse_autocorr(bvp_signal, fps, dominant_hr_bpm):
    """
    Measures how periodic the BVP signal is at the dominant cardiac lag.

    Real human pulses are self-similar across heartbeat cycles → high PAC.
    Deepfake rPPG signals are aperiodic noise → low PAC.

    This is especially discriminating for high-resolution, human-looking
    deepfakes where spatial correlation may appear similar to real faces.
    """
    if len(bvp_signal) < int(fps * 3) or dominant_hr_bpm <= 0:
        return 0.0
    lag = int(round(fps / (dominant_hr_bpm / 60.0)))
    if lag <= 0 or lag >= len(bvp_signal):
        return 0.0
    x = bvp_signal - np.mean(bvp_signal)
    var = np.var(x) + 1e-12
    ac = np.correlate(x, x, mode='full')[len(x) - 1:]
    ac /= (var * len(x))
    margin = max(2, int(0.15 * lag))
    lo = max(1, lag - margin)
    hi = min(len(ac) - 1, lag + margin)
    if lo > hi:
        return float(ac[lag]) if lag < len(ac) else 0.0
    return float(np.max(ac[lo:hi + 1]))


# ────────────────────────────────────────────────────────────────────────────
# Feature 5 — Spectral SNR (in-band peak-to-noise ratio)
# ────────────────────────────────────────────────────────────────────────────
def compute_spectral_snr(bvp_signal, fps):
    """
    Ratio of spectral peak power to mean in-band power.

    A real biological pulse creates a sharp, dominant spectral spike (SNR > 2.5).
    A low-resolution or noisy deepfake has a flat, broadband spectrum (SNR ~ 1).
    """
    if len(bvp_signal) < 4:
        return 1.0
    nperseg = min(len(bvp_signal), int(fps * 5.0))
    freqs, psd = welch(bvp_signal, fs=fps, nperseg=nperseg, nfft=2048)
    mask = (freqs >= 0.75) & (freqs <= 2.5)
    if not np.any(mask):
        return 1.0
    peak = float(psd[mask].max())
    mean = float(psd[mask].mean()) + 1e-12
    return peak / mean


# ────────────────────────────────────────────────────────────────────────────
# Master Composite Score
# ────────────────────────────────────────────────────────────────────────────
def compute_biological_anomaly_score(
    spatial_corr,
    hr_diff,
    sdnn=0.0,
    rmssd=0.0,
    sdsd=0.0,
    bvp_r5=None,
    fps=30.0,
    hr_center=75.0,
    face_dim_px=200,
):
    """
    5-feature composite biological anomaly score [0.0, 1.0].

    Designed to handle:
    1. Human-looking deepfakes (high quality face swaps, high spatial_corr)
       → Detected via Pulse Autocorrelation (PAC) deficit + Spectral SNR
    2. Low-resolution deepfakes (face <= 80px)
       → Detected via SNR collapse + SDNN flatline, not spatial_corr
    3. Standard face swaps
       → Detected via Spatial Coupling + Regional HR divergence

    SCORE < 0.40 → REAL
    SCORE >= 0.40 → DEEPFAKE
    """

    # ── F1: Synthetic Mask Coupling
    # Real: independent micro-variation → low corr (0.05–0.35)
    # Deepfake: 2D texture coupling → high corr (0.40–0.92)
    synthetic_coupling = float(np.clip((spatial_corr - 0.30) / 0.55, 0.0, 1.0))

    # ── F2: Regional Frequency Divergence
    # Real: one heart → all zones same BPM (diff < 8 BPM)
    # Deepfake with seam: inner/outer zones lock on different noise freqs (> 15 BPM)
    freq_divergence = float(np.clip((hr_diff - 8.0) / 30.0, 0.0, 1.0))

    # ── F3: Pulse Autocorrelation Deficit
    # Real heartbeat: periodic and self-similar → PAC > 0.25
    # GAN/swap face: aperiodic noise → PAC < 0.15
    # Catches human-looking deepfakes where spatial_corr alone fails
    pac = 0.0
    if bvp_r5 is not None and len(bvp_r5) > int(fps * 3):
        pac = compute_pulse_autocorr(bvp_r5, fps, hr_center)
    pac_deficit = float(np.clip(1.0 - (pac - 0.10) / 0.40, 0.0, 1.0))

    # ── F4: Spectral SNR anomaly
    # Real skin: sharp cardiac spectral spike → SNR > 2.5
    # Low-res deepfake (<80px): flat broadband noise → SNR ≈ 1.0–2.0
    snr = 1.0
    if bvp_r5 is not None and len(bvp_r5) > 4:
        snr = compute_spectral_snr(bvp_r5, fps)
    snr_anomaly = float(np.clip((3.0 - snr) / 1.5, 0.0, 1.0))

    # ── F5: HRV Anomaly
    # Real resting HRV SDNN: 5–20 BPM
    # Flatline (< 5): deepfake locked to one spectral bin (too few STFT windows, very still face)
    # Extreme noise (> 20): heavy compression / extreme motion blur
    if 5.0 <= sdnn <= 20.0:
        hrv_anomaly = 0.0
    elif sdnn < 5.0:
        # Sharpened penalty: span 0→1 over [5.0 → 2.0], so SDNN=1.76 → hrv=1.0
        hrv_anomaly = float(np.clip((5.0 - sdnn) / 3.0, 0.0, 1.0))
    else:
        hrv_anomaly = float(np.clip((sdnn - 20.0) / 10.0, 0.0, 1.0))

    # ── Adaptive weights by face resolution
    # Low-res faces (<= 80px): spatial_corr is unreliable — only a few hundred pixels
    # per ROI make Pearson correlation highly sensitive to JPEG block artefacts.
    # For these, shift weight heavily onto HRV flatline + frequency divergence + PAC.
    # Normal-res faces: spatial coupling is the most reliable primary discriminator.
    is_low_res = face_dim_px <= 80
    if is_low_res:
        w = dict(coupling=0.05, freq_div=0.30, pac=0.20, snr=0.10, hrv=0.35)
    else:
        w = dict(coupling=0.40, freq_div=0.20, pac=0.20, snr=0.10, hrv=0.10)

    score = (
        w['coupling'] * synthetic_coupling +
        w['freq_div'] * freq_divergence +
        w['pac']      * pac_deficit +
        w['snr']      * snr_anomaly +
        w['hrv']      * hrv_anomaly
    )
    return float(np.clip(score, 0.0, 1.0))
