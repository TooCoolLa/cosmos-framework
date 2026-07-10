import os
import shutil
import json

def main():
    all_frames_dir = '/cloud/cloud-ssd1/videos/all_frames_480p'
    # 使用独立的子文件夹避免与 180 帧冲突
    output_dir = '/cloud/cloud-ssd1/videos/chunks_480p_b60o12'
    jsonl_output_path = '/cloud/cloud-ssd1/av_batch_480p_b60o12.jsonl'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Scanning extracted frame directory {all_frames_dir}...")
    if not os.path.exists(all_frames_dir):
        print(f"Error: Directory {all_frames_dir} does not exist!")
        return
        
    frames = sorted([f for f in os.listdir(all_frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_frames = len(frames)
    print(f"Total extracted frames found: {total_frames}")
    
    if total_frames == 0:
        return
        
    # b60o12 配置
    chunk_len = 60
    overlap = 12
    stride = chunk_len - overlap
    
    # 算出所有的 chunk 的起止帧索引 (0-indexed)
    chunks_indices = []
    i = 0
    while True:
        start_idx = i * stride
        end_idx = start_idx + chunk_len
        if end_idx >= total_frames:
            start_idx = max(0, total_frames - chunk_len)
            end_idx = total_frames
            chunks_indices.append((start_idx, end_idx))
            break
        chunks_indices.append((start_idx, end_idx))
        i += 1
        
    print(f"Generating hardlinks/symlinks for {len(chunks_indices)} chunks of b60o12...")
    jsonl_records = []
    
    for chunk_idx, (start, end) in enumerate(chunks_indices):
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}")
        os.makedirs(chunk_path, exist_ok=True)
        
        # 硬链接快速映射
        for frame_offset, frame_global_idx in enumerate(range(start, end)):
            src_name = frames[frame_global_idx]
            dst_name = f"frame_{frame_offset:05d}.jpg"
            src_file = os.path.join(all_frames_dir, src_name)
            dst_file = os.path.join(chunk_path, dst_name)
            
            if os.path.exists(dst_file):
                os.remove(dst_file)
                
            try:
                os.link(src_file, dst_file)
            except OSError:
                try:
                    os.symlink(src_file, dst_file)
                except OSError:
                    shutil.copy2(src_file, dst_file)
                    
        # 写入 b60o12 的 JSONL 记录
        first_frame_path = os.path.join(chunk_path, "frame_00000.jpg")
        record = {
            "action_chunk_size": 60,  # 设为 60 帧动作长度
            "domain_name": "av",
            "fps": 30,
            "view_point": "ego_view",
            "guidance": 1.0,
            "image_size": 480,
            "model_mode": "inverse_dynamics",
            "name": f"batch_{chunk_idx}_{start}_{end}",
            "num_steps": 50,
            "prompt": "You are an autonomous vehicle driving on a road. The camera is mounted on the front windshield facing forward.",
            "seed": 0,
            "shift": 10.0,
            "vision_path": first_frame_path
        }
        jsonl_records.append(record)
        
        if (chunk_idx + 1) % 100 == 0 or chunk_idx == len(chunks_indices) - 1:
            print(f"Mapped {chunk_idx + 1}/{len(chunks_indices)} chunks...")
            
    # 写入总 JSONL
    with open(jsonl_output_path, 'w', encoding='utf-8') as f:
        for r in jsonl_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    print(f"Successfully processed video into {len(chunks_indices)} chunk directories for b60o12.")
    print(f"JSONL config file saved to {jsonl_output_path}")

if __name__ == '__main__':
    main()
