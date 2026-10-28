import cv2
import numpy as np
from src.roi_detector import extract_r5_roi, compute_skin_rgb_mean

def extract_temporal_rgb(frame_paths, sample_rate_step=1):
    rgb_series = []
    for i in range(0, len(frame_paths), sample_rate_step):
        frame = cv2.imread(frame_paths[i])
        if frame is None:
            continue
            
        r5_crop = extract_r5_roi(frame)
        rgb_mean = compute_skin_rgb_mean(r5_crop)
        if rgb_mean is not None:
            rgb_series.append(rgb_mean)
        
    return np.array(rgb_series, dtype=np.float64) if rgb_series else np.empty((0, 3))