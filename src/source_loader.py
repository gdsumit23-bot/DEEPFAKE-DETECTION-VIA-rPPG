import os
import re
from collections import defaultdict

FRAME_PATTERN = re.compile(r"^(.*)_f(\d+)\.(jpg|png|jpeg)$", re.IGNORECASE)

def load_video_sequences_from_folders(frames_dir, min_frames=4):
    sequences = defaultdict(lambda: {'frames': [], 'label': None, 'is_fake': None})
    total_images_found = 0
    
    for root, _, files in os.walk(frames_dir):
        if not files:
            continue
        folder_name = os.path.basename(root)
        is_fake = 0 if folder_name.lower() == 'original' else 1
        
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                total_images_found += 1
                full_path = os.path.join(root, file)
                
                match = FRAME_PATTERN.match(file)
                if match:
                    video_id = match.group(1)
                    frame_idx = int(match.group(2))
                else:
                    video_id = os.path.splitext(file)[0]
                    frame_idx = 0
                    
                unique_vid_id = f"{folder_name}_{video_id}"
                seq = sequences[unique_vid_id]
                seq['frames'].append((frame_idx, full_path))
                if seq['label'] is None:
                    seq['label'] = folder_name
                    seq['is_fake'] = is_fake

    print(f"--> DEBUG: Found {total_images_found} total image files across all folders.")

    processed_sequences = {}
    for vid, data in sequences.items():
        if len(data['frames']) >= min_frames:
            data['frames'].sort(key=lambda x: x[0])
            processed_sequences[vid] = {
                'frame_paths': [f[1] for f in data['frames']],
                'label': data['label'],
                'is_fake': data['is_fake']
            }

    return processed_sequences