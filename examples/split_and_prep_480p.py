import os
import cv2
import json
from collections import deque

def main():
    video_path = '/cloud/cloud-ssd1/videos/MerachVideo10020_fixed.mp4'
    output_dir = '/cloud/cloud-ssd1/videos/chunks_480p'
    jsonl_output_path = '/cloud/cloud-ssd1/av_batch_480p.jsonl'
    jsonl_part1_path = '/cloud/cloud-ssd1/av_batch_480p_part1.jsonl'
    jsonl_part2_path = '/cloud/cloud-ssd1/av_batch_480p_part2.jsonl'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Opening video from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Original video: {width_orig}x{height_orig}, {fps} fps, {total_frames} frames")
    
    chunk_len = 180
    overlap = 20
    stride = chunk_len - overlap
    
    # 算出所有的 chunk 的写出帧索引 (0-indexed)
    write_points = {}
    i = 0
    while True:
        end_frame = i * stride + chunk_len
        if end_frame >= total_frames:
            write_points[total_frames - 1] = "last"
            break
        write_points[end_frame - 1] = i
        i += 1
        
    print(f"Planned chunks: {len(write_points)} (regular stride up to index {i - 1}, plus 1 final overlapping chunk)")
    
    frame_window = deque(maxlen=chunk_len)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width_new, height_new = 854, 480  # 480p 16:9
    
    jsonl_records = []
    
    frame_idx = 0
    actual_chunk_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame_resized = cv2.resize(frame, (width_new, height_new))
        frame_window.append(frame_resized)
        
        if frame_idx in write_points:
            if len(frame_window) == chunk_len:
                out_path = os.path.join(output_dir, f"chunk_{actual_chunk_count:04d}.mp4")
                out = cv2.VideoWriter(out_path, fourcc, 30.0, (width_new, height_new))
                for f in frame_window:
                    out.write(f)
                out.release()
                
                record = {
                    "action_chunk_size": 180,
                    "domain_name": "av",
                    "fps": 30,
                    "view_point": "ego_view",
                    "guidance": 1.0,
                    "image_size": 480,
                    "model_mode": "inverse_dynamics",
                    "name": f"av_chunk_480p_{actual_chunk_count:04d}",
                    "num_steps": 50,
                    "prompt": "You are an autonomous vehicle driving on a road. The camera is mounted on the front windshield facing forward.",
                    "seed": 0,
                    "shift": 10.0,
                    "vision_path": out_path
                }
                jsonl_records.append(record)
                actual_chunk_count += 1
                
        frame_idx += 1
        if frame_idx % 2000 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")
            
    cap.release()
    
    # 均分 JSONL 记录到两个部分
    mid = len(jsonl_records) // 2
    part1_records = jsonl_records[:mid]
    part2_records = jsonl_records[mid:]
    
    # 写入总 JSONL
    with open(jsonl_output_path, 'w', encoding='utf-8') as f:
        for r in jsonl_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    # 写入 Part 1
    with open(jsonl_part1_path, 'w', encoding='utf-8') as f:
        for r in part1_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    # 写入 Part 2
    with open(jsonl_part2_path, 'w', encoding='utf-8') as f:
        for r in part2_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    print(f"Successfully processed video into {actual_chunk_count} chunks.")
    print(f"JSONL part1: {len(part1_records)} records saved to {jsonl_part1_path}")
    print(f"JSONL part2: {len(part2_records)} records saved to {jsonl_part2_path}")

if __name__ == '__main__':
    main()
