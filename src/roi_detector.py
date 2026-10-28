import cv2
import numpy as np

def extract_r5_roi(image):
    """
    R5 corresponds to the central grid [row 1, col 1] in a 3x3 face crop.
    Captures the central nose region (highest significance in Lee et al.: t = 5.216).
    """
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    return image[h // 3 : 2 * (h // 3), w // 3 : 2 * (w // 3)]

def extract_forehead_roi(image):
    """
    Extracts bilateral cheek reference zones (R4 and R6 in the 3x3 grid,
    trimmed inward to exclude background edges) instead of the hair-occluded R2 forehead.
    """
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    y1, y2 = h // 3, 2 * (h // 3)
    
    r4_inner = image[y1:y2, int(0.12 * w) : w // 3]
    r6_inner = image[y1:y2, 2 * (w // 3) : int(0.88 * w)]
    
    if r4_inner.size == 0 or r6_inner.size == 0:
        return image[y1:y2, :]
    return np.hstack((r4_inner, r6_inner))

def compute_skin_rgb_mean(bgr_roi):
    """
    Computes spatial mean [R, G, B] over valid skin pixels using YCrCb masking
    to reject hair, glasses, and background pixels.
    """
    if bgr_roi is None or bgr_roi.size == 0:
        return None
        
    ycrcb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    
    if cv2.countNonZero(skin_mask) > 0.25 * (bgr_roi.shape[0] * bgr_roi.shape[1]):
        b, g, r, _ = cv2.mean(bgr_roi, mask=skin_mask)
    else:
        b, g, r, _ = cv2.mean(bgr_roi)
        
    return [float(r), float(g), float(b)]