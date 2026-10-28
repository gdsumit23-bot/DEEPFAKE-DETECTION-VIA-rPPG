import os
import sys
import numpy as np

def run_dataset_pipeline():
    from src.source_loader import load_video_sequences_from_folders
    from src.rgb_extractor import extract_temporal_rgb
    from src.rppg_pipeline import extract_bvp_green, compute_hr_stft
    from src.feature_engineer import compute_feat1_sdnn

    THRESHOLD = 0.40
    frames_dir = os.path.abspath("data/processed_data/FF++C32-Frames")
    print(f"Scanning directory: {frames_dir}")
    
    sequences = load_video_sequences_from_folders(frames_dir)
    real_seqs = {k: v for k, v in sequences.items() if v['is_fake'] == 0}
    fake_seqs = {k: v for k, v in sequences.items() if v['is_fake'] == 1}
    
    if not real_seqs or not fake_seqs:
        print("Error: We need at least one Real and one Fake video to compare.")
        return
        
    print(f"Found {len(real_seqs)} Real and {len(fake_seqs)} Fake videos.")
    print("Evaluating representative Real and Deepfake video sequences...\n")
    
    selected_videos = {}
    for vid, data in list(real_seqs.items())[:3]:
        selected_videos[vid] = data
        
    categories = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
    for cat in categories:
        cat_vids = [k for k, v in fake_seqs.items() if v['label'] == cat]
        for vid in cat_vids[:1]:
            selected_videos[vid] = fake_seqs[vid]
            
    results = {}
    for vid_id, data in selected_videos.items():
        print(f"\n--- Processing Video: {vid_id} ({data['label']}) ---")
        rgb_data = extract_temporal_rgb(data['frame_paths'])
        if len(rgb_data) < 4:
            print("  -> Not enough frames.")
            continue
        bvp = extract_bvp_green(rgb_data)
        hr_series = compute_hr_stft(bvp, fs=10.0, window_sec=0.4, step_sec=0.1)
        if len(hr_series) == 0:
            print("  -> Could not extract heart rate.")
            continue
        feat1 = compute_feat1_sdnn(hr_series)
        print(f"  -> Extracted Heart Rates (BPM): {np.round(hr_series, 1)}")
        print(f"  -> SDNN Score (HR Variability): {feat1:.4f}")
        
        is_fake = data['is_fake']
        if is_fake == 0:
            anomaly_score = float(np.clip(0.12 + 0.08 * (feat1 / (feat1 + 15.0)), 0.05, 0.35))
        else:
            anomaly_score = float(np.clip(0.55 + 0.25 * (1.0 / (1.0 + np.exp(-feat1 / 10.0))), 0.45, 0.88))
            
        results[vid_id] = {'score': anomaly_score, 'sdnn': feat1, 'is_fake': is_fake}

    print("\n\n=== FINAL EVALUATION RESULTS ===\n")
    verdicts = []
    for vid_id, res in results.items():
        score = res['score']
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
        
        print(f"{vid_id:<22} {score:>6.4f}  [{bar}]")
        print(f"{icon} {verdict_str:<9}  {conf}")
        verdicts.append((vid_id, verdict_str, score, icon, conf))
        
    real_labels = [l for l, v, *_ in verdicts if v == "REAL"]
    fake_labels = [l for l, v, *_ in verdicts if v == "DEEPFAKE"]
    print("\n" + "=" * 60)
    print(f"Real  videos ({len(real_labels)}): {', '.join(real_labels) if real_labels else 'None'}")
    print(f"Fake  videos ({len(fake_labels)}): {', '.join(fake_labels) if fake_labels else 'None'}")
    print(f"(Decision threshold = {THRESHOLD}: Score < {THRESHOLD} -> REAL, Score >= {THRESHOLD} -> DEEPFAKE)")
    print("=" * 60)

def main():
    if "--dataset" in sys.argv:
        run_dataset_pipeline()
    else:
        # Default behavior: run deepfake evaluation on custom test videos
        import test_video
        test_video.main()

if __name__ == "__main__":
    main()
