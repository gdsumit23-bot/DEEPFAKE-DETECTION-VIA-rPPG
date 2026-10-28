import os
import sys
import cv2
import numpy as np

from src.roi_detector import extract_r5_roi, extract_forehead_roi, compute_skin_rgb_mean
from src.rppg_pipeline import (
    extract_bvp_green,
    extract_bvp_hybrid,
    compute_hr_stft,
    compute_hr_welch,
)
from src.feature_engineer import (
    compute_feat1_sdnn,
    compute_feat2_rmssd,
    compute_feat3_sdsd,
    compute_biological_anomaly_score,
    compute_spectral_snr,
    compute_pulse_autocorr,
)

# ──────────────────────────────────────────────────────────────────────────────
# Decision Threshold
# Real faces:   score < 0.40
# Deepfakes:    score >= 0.40
# ──────────────────────────────────────────────────────────────────────────────
THRESHOLD = 0.40
DETECT_INTERVAL = 5   # Run Haar detector every N frames for speed


def process_mp4(video_path):
    video_name = os.path.basename(video_path)
    print(f"\n--- Analyzing Custom Video: {video_name} ---")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    fps = float(fps)
    print(f"Video FPS detected: {fps:.2f}")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    current_box = None
    r5_rgb, fh_rgb = [], []
    face_sizes = []
    frame_count = 0

    print("Extracting and dynamically tracking faces (Haar + EMA Smoothed)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if current_box is None or frame_count % DETECT_INTERVAL == 0:
            scale = min(1.0, 480.0 / max(h, w))
            small_gray = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), fx=scale, fy=scale
            )
            faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                target_box = max(faces, key=lambda f: f[2] * f[3])
                t_orig = [
                    target_box[0] / scale, target_box[1] / scale,
                    target_box[2] / scale, target_box[3] / scale,
                ]
                if current_box is None:
                    current_box = t_orig
                else:
                    current_box = [0.2 * t + 0.8 * c for t, c in zip(t_orig, current_box)]

        frame_count += 1
        if current_box is None:
            continue

        x, y, bw, bh = [int(v) for v in current_box]
        x, y = max(0, x), max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        if bw < 20 or bh < 20:
            continue

        face_sizes.append(int((bw * bh) ** 0.5))   # √area ≈ effective face dimension
        face_crop = frame[y: y + bh, x: x + bw]

        r5_mean = compute_skin_rgb_mean(extract_r5_roi(face_crop))
        fh_mean = compute_skin_rgb_mean(extract_forehead_roi(face_crop))
        if r5_mean is not None and fh_mean is not None:
            r5_rgb.append(r5_mean)
            fh_rgb.append(fh_mean)

    cap.release()

    if len(r5_rgb) < int(fps * 3):
        print(f"Error: Only {len(r5_rgb)} face frames (need {int(fps * 3)}).")
        return None

    r5_rgb = np.array(r5_rgb, dtype=np.float64)
    fh_rgb = np.array(fh_rgb, dtype=np.float64)
    mean_face_dim = int(np.median(face_sizes)) if face_sizes else 200

    print(f"  Face frames: {len(r5_rgb)} | Mean face dim: {mean_face_dim}px "
          f"({'LOW-RES' if mean_face_dim <= 80 else 'normal'})")

    print("Analyzing Hemodynamics & Biological Consistency...")
    bvp_r5     = extract_bvp_hybrid(r5_rgb, fps=fps)
    bvp_fh     = extract_bvp_hybrid(fh_rgb, fps=fps)
    bvp_r5_green = extract_bvp_green(r5_rgb, fps=fps)

    # ── Regional dominant heart rates
    hr_center   = compute_hr_welch(bvp_r5, fs=fps)
    hr_forehead = compute_hr_welch(bvp_fh, fs=fps)
    hr_diff     = abs(hr_forehead - hr_center)

    # ── Spatial hemodynamic coupling
    min_len     = min(len(bvp_r5), len(bvp_fh))
    spatial_corr = float(np.corrcoef(bvp_r5[:min_len], bvp_fh[:min_len])[0, 1])
    if np.isnan(spatial_corr):
        spatial_corr = 0.0

    # ── Temporal HRV features
    hr_series = compute_hr_stft(bvp_r5_green, fs=fps, window_sec=4.0, step_sec=0.1)
    sdnn  = compute_feat1_sdnn(hr_series)
    rmssd = compute_feat2_rmssd(hr_series)
    sdsd  = compute_feat3_sdsd(hr_series)

    # ── Pulse periodicity and spectral quality
    pac = compute_pulse_autocorr(bvp_r5, fps, hr_center)
    snr = compute_spectral_snr(bvp_r5, fps)

    # ── Composite biological anomaly score (5-feature)
    anomaly_score = compute_biological_anomaly_score(
        spatial_corr=spatial_corr,
        hr_diff=hr_diff,
        sdnn=sdnn,
        rmssd=rmssd,
        sdsd=sdsd,
        bvp_r5=bvp_r5,
        fps=fps,
        hr_center=hr_center,
        face_dim_px=mean_face_dim,
    )

    preview_hr = np.round(hr_series[::10], 1) if len(hr_series) else [round(hr_center, 1)]
    print(f"  -> HR series (1s steps):         {preview_hr}")
    print(f"  -> Center HR: {hr_center:.1f} BPM  |  Reference HR: {hr_forehead:.1f} BPM"
          f"  (Spread: {hr_diff:.1f} BPM)")
    print(f"  -> Spatial Mask Coupling:        {spatial_corr:.4f}  "
          f"({'HIGH – suspicious' if spatial_corr > 0.55 else 'normal'})")
    print(f"  -> Pulse Autocorrelation (PAC):  {pac:.4f}  "
          f"({'weak – suspicious' if pac < 0.20 else 'strong – real-like'})")
    print(f"  -> Spectral SNR:                 {snr:.2f}  "
          f"({'low – suspicious' if snr < 2.0 else 'good'})")
    print(f"  -> SDNN: {sdnn:.2f}  RMSSD: {rmssd:.2f}  SDSD: {sdsd:.2f} BPM")
    print(f"  -> Biological Anomaly Score:     {anomaly_score:.4f}")

    return {
        'score':        anomaly_score,
        'spatial_corr': spatial_corr,
        'hr_diff':      hr_diff,
        'hr_center':    hr_center,
        'hr_forehead':  hr_forehead,
        'sdnn':         sdnn,
        'rmssd':        rmssd,
        'sdsd':         sdsd,
        'pac':          pac,
        'snr':          snr,
        'face_dim_px':  mean_face_dim,
    }


def main(custom_videos=None):
    if custom_videos is not None:
        videos = custom_videos
    elif len(sys.argv) > 1:
        videos = [(os.path.basename(p), p) for p in sys.argv[1:] if not p.startswith("-")]
        if not videos:
            videos = [
                ("VIDEO 1", "data/video1.mp4"),
                ("VIDEO 2", "data/video2.mp4"),
                ("VIDEO 3", "data/video3.mp4"),
                ("VIDEO 4", "data/video4.mp4"),
                ("VIDEO 5", "data/video5.mp4"),
                ("VIDEO 6", "data/video6.mp4"),
            ]
    else:
        videos = [
            ("VIDEO 1", "data/video1.mp4"),
            ("VIDEO 2", "data/video2.mp4"),
            ("VIDEO 3", "data/video3.mp4"),
            ("VIDEO 4", "data/video4.mp4"),
            ("VIDEO 5", "data/video5.mp4"),
            ("VIDEO 6", "data/video6.mp4"),
        ]

    print("=== DEEPFAKE DETECTOR: HIGH ACCURACY MODE ===")
    results = {}
    for label, path in videos:
        if os.path.exists(path):
            results[label] = process_mp4(path)
        else:
            print(f"File not found: {path}")

    print("\n" + "=" * 60)
    print("                    FINAL EVALUATION RESULTS")
    print("=" * 60 + "\n")

    verdicts = []
    for label, res in results.items():
        if res is not None:
            score   = res['score']
            is_fake = score >= THRESHOLD
            verdict_str = "DEEPFAKE" if is_fake else "REAL"
            icon        = "🚨" if is_fake else "✅"
            conf_margin = abs(score - THRESHOLD)
            if conf_margin > 0.20:
                conf = "High confidence"
            elif conf_margin > 0.08:
                conf = "Medium confidence"
            else:
                conf = "Low confidence"
            
            bar_s = int(score * 20)
            bar_s = max(0, min(20, bar_s))
            bar   = "█" * bar_s + "░" * (20 - bar_s)

            print(f"{label:<12} {score:>6.4f}  [{bar}]")
            print(f"{icon} {verdict_str:<9}  {conf}")
            verdicts.append((label, verdict_str, score, icon, conf))
        else:
            print(f"{label:<12} {'N/A':>6}")
            print(f"⚠️  ERROR      Insufficient face frames")
            verdicts.append((label, "ERROR", None, "⚠️", ""))

    real_labels = [l for l, v, *_ in verdicts if v == "REAL"]
    fake_labels = [l for l, v, *_ in verdicts if v == "DEEPFAKE"]
    print("\n" + "=" * 60)
    print(f"Real  videos ({len(real_labels)}): {', '.join(real_labels) if real_labels else 'None'}")
    print(f"Fake  videos ({len(fake_labels)}): {', '.join(fake_labels) if fake_labels else 'None'}")
    print(f"(Decision threshold = {THRESHOLD}: Score < {THRESHOLD} -> REAL, Score >= {THRESHOLD} -> DEEPFAKE)")
    print("=" * 60)


if __name__ == "__main__":
    main()
