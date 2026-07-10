#!/usr/bin/env python3
"""
Trajectory Merger and C4D Processor
----------------------------------
This script:
1. Loads 9D trajectory slices [pos(3), rot_6d(6)] from all batch inference JSONs.
2. Restores orthogonal C2W matrices from the 6D rotation representation.
3. Aligns overlapping blocks using SE3 rigid alignment over the overlap margins 
   (excluding the noisy head/tail 10 frames of each batch segment).
4. Cascades the segments into a single continuous right-handed trajectory.
5. Saves the right-handed trajectory as a 7D numpy array [pos(3), quat_xyzw(4)] (optional).
6. Converts the merged trajectory into the C4D Left-Handed coordinate space.
7. Aligns the final C4D trajectory such that:
   - Frame 0 is located at the origin with identity rotation.
   - The camera displacement over the first 2 seconds is aligned parallel to the positive Z-axis.
8. Outputs the final JSON importable into Cinema 4D.
"""

import os
import json
import glob
import re
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot

def get_alignment_rotation(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.eye(3)
    v = v / norm
    u = np.array([0.0, 0.0, 1.0])
    
    c = v[2] # dot product
    if c < -0.99999:
        # 180 degree rotation around X-axis
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
    
    s_x = v[1]
    s_y = -v[0]
    
    skew = np.array([
        [0.0, 0.0, s_y],
        [0.0, 0.0, -s_x],
        [-s_y, s_x, 0.0]
    ])
    
    R = np.eye(3) + skew + skew @ skew / (1.0 + c)
    return R

def rot6d_to_matrix(rot_6d):
    """
    Gram-Schmidt orthogonalization to convert 6D rotation vectors to 3x3 rotation matrices.
    rot_6d: [N, 6] array, where columns 0:3 is x_raw, and columns 3:6 is y_raw.
    """
    x_raw = rot_6d[:, :3]
    y_raw = rot_6d[:, 3:]
    
    x = x_raw / (np.linalg.norm(x_raw, axis=1, keepdims=True) + 1e-9)
    # Make y orthogonal to x
    dot_xy = np.sum(x * y_raw, axis=1, keepdims=True)
    y_ortho = y_raw - dot_xy * x
    y = y_ortho / (np.linalg.norm(y_ortho, axis=1, keepdims=True) + 1e-9)
    
    z = np.cross(x, y)
    
    # Pack into [N, 3, 3] rotation matrices
    Rs = np.stack([x, y, z], axis=2)
    return Rs

def average_se3_transforms(transforms):
    """
    Calculates the average SE3 transform from a list of 4x4 matrices.
    """
    n = len(transforms)
    if n == 0:
        return np.eye(4)
    if n == 1:
        return transforms[0]
        
    translations = [T[:3, 3] for T in transforms]
    rotations = [T[:3, :3] for T in transforms]
    
    avg_t = np.mean(translations, axis=0)
    
    # Average rotation using quaternions
    quats = []
    for R in rotations:
        q = Rot.from_matrix(R).as_quat() # [x, y, z, w]
        # Hemisphere alignment
        if len(quats) > 0 and np.dot(quats[0], q) < 0:
            q = -q
        quats.append(q)
        
    avg_q = np.mean(quats, axis=0)
    avg_q = avg_q / (np.linalg.norm(avg_q) + 1e-9)
    avg_R = Rot.from_quat(avg_q).as_matrix()
    
    T_avg = np.eye(4)
    T_avg[:3, :3] = avg_R
    T_avg[:3, 3] = avg_t
    return T_avg

def parse_chunk_idx(dir_name):
    match = re.search(r'chunk_(\d+)', dir_name)
    return int(match.group(1)) if match else -1

def merge_and_align_trajectories(input_dir, chunk_len=180, overlap=20):
    print(f"[*] Scanning for batch data in: {input_dir}")
    
    # Try flat batch_*.json files first
    flat_files = glob.glob(os.path.join(input_dir, "batch_*.json"))
    
    chunks_poses = []
    stride = chunk_len - overlap
    
    if flat_files:
        print(f"[*] Found {len(flat_files)} flat batch JSON files. Sorting...")
        def parse_flat_batch_idx(file_path):
            name = os.path.basename(file_path)
            match = re.search(r'batch_(\d+)_', name)
            return int(match.group(1)) if match else -1
            
        flat_files = sorted(flat_files, key=parse_flat_batch_idx)
        
        for idx, fpath in enumerate(flat_files):
            name = os.path.basename(fpath)
            match = re.search(r'batch_\d+_(\d+)_(\d+)\.json', name)
            if not match:
                print(f"[!] Warning: Cannot parse frame bounds from filename {name}, skipping.")
                continue
            start_frame = int(match.group(1))
            end_frame = int(match.group(2))
            
            with open(fpath, 'r') as f:
                js_data = json.load(f)
                
            action_list = js_data["outputs"][0]["content"]["action"]
            actions = np.array(action_list)
            
            n_frames = len(actions)
            Cs = actions[:, :3]
            Rs = rot6d_to_matrix(actions[:, 3:])
            
            T_c2w = np.zeros((n_frames, 4, 4))
            for f in range(n_frames):
                T_c2w[f, :3, :3] = Rs[f]
                T_c2w[f, :3, 3] = Cs[f]
                T_c2w[f, 3, 3] = 1.0
                
            chunks_poses.append({
                "idx": idx,
                "poses": T_c2w,
                "start_frame": start_frame,
                "end_frame": end_frame
            })
    else:
        subdirs = sorted([
            d for d in glob.glob(os.path.join(input_dir, "av_chunk_*"))
            if os.path.isdir(d)
        ], key=parse_chunk_idx)
        
        if not subdirs:
            raise FileNotFoundError(f"No batch_*.json or av_chunk_* directories found in {input_dir}")
            
        print(f"[*] Found {len(subdirs)} av_chunk_* subdirs. Loading sample_outputs.json...")
        for idx, sdir in enumerate(subdirs):
            json_path = os.path.join(sdir, "sample_outputs.json")
            if not os.path.exists(json_path):
                print(f"[!] Warning: sample_outputs.json not found in {sdir}, skipping.")
                continue
                
            with open(json_path, 'r') as f:
                js_data = json.load(f)
                
            action_list = js_data["outputs"][0]["content"]["action"]
            actions = np.array(action_list)
            
            n_frames = len(actions)
            Cs = actions[:, :3]
            Rs = rot6d_to_matrix(actions[:, 3:])
            
            T_c2w = np.zeros((n_frames, 4, 4))
            for f in range(n_frames):
                T_c2w[f, :3, :3] = Rs[f]
                T_c2w[f, :3, 3] = Cs[f]
                T_c2w[f, 3, 3] = 1.0
                
            chunks_poses.append({
                "idx": idx,
                "poses": T_c2w,
                "start_frame": idx * stride,
                "end_frame": idx * stride + n_frames
            })
            
    if not chunks_poses:
        raise ValueError("No valid trajectory chunks were loaded.")
        
    # Determine the total trajectory length
    total_frames = chunks_poses[-1]["end_frame"]
    print(f"[*] Total frames in sequence: {total_frames}")
    
    # Initialize the merged trajectory dictionary. Key is global frame index.
    # We will accumulate aligned poses per frame.
    aligned_poses_per_frame = {g: [] for g in range(total_frames)}
    
    # We cascade-align each chunk to the cumulative trajectory.
    # Chunk 0 acts as the reference frame (untransformed).
    chunk0 = chunks_poses[0]
    for f_local in range(len(chunk0["poses"])):
        g_idx = chunk0["start_frame"] + f_local
        # Margin filtering: discard head and tail 10 frames
        if 10 <= f_local < len(chunk0["poses"]) - 10:
            aligned_poses_per_frame[g_idx].append(chunk0["poses"][f_local])
            
    # For subsequent chunks, we calculate an aligning SE3 transform based on the overlap
    # regions inside their valid ranges [10, chunk_len - 10]
    for i in range(1, len(chunks_poses)):
        curr_chunk = chunks_poses[i]
        curr_poses = curr_chunk["poses"]
        c_len = len(curr_poses)
        
        # Define current chunk's valid global frame indices
        valid_indices_curr = set(range(curr_chunk["start_frame"] + 10, curr_chunk["start_frame"] + c_len - 10))
        
        # Find global overlap frames with the already merged trajectory valid parts
        already_merged_indices = set([g for g, p in aligned_poses_per_frame.items() if len(p) > 0])
        overlap_global_frames = sorted(list(valid_indices_curr.intersection(already_merged_indices)))
        
        T_align = None
        if overlap_global_frames:
            print(f"[*] Aligning chunk {curr_chunk['idx']} on {len(overlap_global_frames)} overlapping frames (valid margins)...")
            transforms_to_avg = []
            for g in overlap_global_frames:
                local_idx_curr = g - curr_chunk["start_frame"]
                # Average pose of the already merged segment at frame g
                T_ref = average_se3_transforms(aligned_poses_per_frame[g])
                T_curr = curr_poses[local_idx_curr]
                
                # T_align_g @ T_curr = T_ref  => T_align_g = T_ref @ T_curr^-1
                T_align_g = T_ref @ np.linalg.inv(T_curr)
                transforms_to_avg.append(T_align_g)
                
            T_align = average_se3_transforms(transforms_to_avg)
        else:
            # Fallback: if no overlap in valid zones, use the original overlap region without margin margins
            orig_overlap_start = max(curr_chunk["start_frame"], chunks_poses[i-1]["start_frame"])
            orig_overlap_end = min(curr_chunk["end_frame"], chunks_poses[i-1]["end_frame"])
            overlap_global_frames = sorted(list(range(orig_overlap_start, orig_overlap_end)))
            
            # Narrow overlap to center region by shaving off some frames to avoid boundary issues
            if len(overlap_global_frames) > 4:
                overlap_global_frames = overlap_global_frames[2:-2]
                
            print(f"[!] Warning: No valid margin overlap. Falling back to original overlap ({len(overlap_global_frames)} frames)...")
            transforms_to_avg = []
            for g in overlap_global_frames:
                # Find corresponding poses in both chunks directly
                local_idx_prev = g - chunks_poses[i-1]["start_frame"]
                local_idx_curr = g - curr_chunk["start_frame"]
                
                T_ref = chunks_poses[i-1]["poses"][local_idx_prev]
                T_curr = curr_poses[local_idx_curr]
                
                T_align_g = T_ref @ np.linalg.inv(T_curr)
                transforms_to_avg.append(T_align_g)
                
            T_align = average_se3_transforms(transforms_to_avg)
            
        # Apply the computed SE3 transform to the valid part of the current chunk
        for f_local in range(c_len):
            if 10 <= f_local < c_len - 10:
                g_idx = curr_chunk["start_frame"] + f_local
                T_aligned = T_align @ curr_poses[f_local]
                aligned_poses_per_frame[g_idx].append(T_aligned)
                
    # Finalize merged trajectory by averaging overlapping frame poses
    final_trajectory_c2w = []
    valid_global_indices = []
    
    for g in sorted(aligned_poses_per_frame.keys()):
        poses = aligned_poses_per_frame[g]
        if poses:
            T_avg = average_se3_transforms(poses)
            final_trajectory_c2w.append(T_avg)
            valid_global_indices.append(g)
            
    print(f"[*] Trajectory successfully merged: {len(final_trajectory_c2w)} valid frames.")
    return np.array(final_trajectory_c2w), valid_global_indices

def main():
    parser = argparse.ArgumentParser(description="Merge batch trajectory chunks and process for Cinema 4D")
    parser.add_argument("-i", "--input_dir", default="av_final", help="Path to input directory containing av_chunk_*")
    parser.add_argument("-o", "--output_json", default="av_final/merged_c4d.json", help="Path to output JSON file for C4D")
    parser.add_argument("-n", "--output_npy", default="av_final/merged_trajectory.npy", help="Path to save merged 7D numpy trajectory")
    parser.add_argument("-w", "--chunk_len", type=int, default=180, help="Chunk length (default: 180)")
    parser.add_argument("-v", "--overlap", type=int, default=20, help="Overlap size (default: 20)")
    parser.add_argument("--fps", type=float, default=30.0, help="Camera video FPS (default: 30.0)")
    parser.add_argument("-ws", "--window_smooth", type=int, default=0, help="Smoothing window size for velocities (default: 0, no smoothing)")
    
    args = parser.parse_args()
    
    # 1. Merge all trajectory chunks using SE3 alignment
    final_c2w, global_indices = merge_and_align_trajectories(args.input_dir, args.chunk_len, args.overlap)
    num_frames = len(final_c2w)
    
    # Apply velocity-based sliding window smoothing if requested
    if args.window_smooth > 0:
        print(f"[*] Applying velocity-based sliding window smoothing (window={args.window_smooth})...")
        Cs = final_c2w[:, :3, 3]
        velocities = np.diff(Cs, axis=0)
        smoothed_v = np.zeros_like(velocities)
        kernel = np.ones(args.window_smooth) / args.window_smooth
        for i in range(3):
            smoothed_v[:, i] = np.convolve(velocities[:, i], kernel, mode='same')
        for i in range(1, num_frames):
            Cs[i] = Cs[i-1] + smoothed_v[i-1]
        final_c2w[:, :3, 3] = Cs
    
    # 2. Save the merged right-handed trajectory as [N, 7] format (pos(3), quat_xyzw(4))
    if args.output_npy:
        print(f"[*] Saving merged right-handed trajectory to: {args.output_npy}")
        merged_7d = np.zeros((num_frames, 7))
        for i in range(num_frames):
            T = final_c2w[i]
            pos = T[:3, 3]
            quat = Rot.from_matrix(T[:3, :3]).as_quat() # XYZW representation
            merged_7d[i, :3] = pos
            merged_7d[i, 3:] = quat
            
        os.makedirs(os.path.dirname(args.output_npy), exist_ok=True)
        np.save(args.output_npy, merged_7d)
        print("[+] Merged NPY trajectory saved successfully.")
        
    # 3. Convert merged trajectory to Cinema 4D Left-Handed space
    # S flips Y and changes handedness. Z remains forward.
    print("[*] Converting merged trajectory to C4D Left-Handed space...")
    S = np.diag([1, -1, 1])
    
    Rs_c4d = []
    Cs_c4d = []
    for i in range(num_frames):
        R_c2w = final_c2w[i, :3, :3]
        C_c2w = final_c2w[i, :3, 3]
        
        R_c4d = S @ R_c2w @ S
        C_c4d = S @ C_c2w
        
        Rs_c4d.append(R_c4d)
        Cs_c4d.append(C_c4d)
        
    Rs_c4d = np.array(Rs_c4d)
    Cs_c4d = np.array(Cs_c4d)
    
    # 4. Perform Frame 0 Alignment (Origin and Identity Rotation)
    R0 = Rs_c4d[0]
    C0 = Cs_c4d[0]
    
    Rs_rel = []
    Cs_rel = []
    for i in range(num_frames):
        R_rel = R0.T @ Rs_c4d[i]
        C_rel = R0.T @ (Cs_c4d[i] - C0)
        Rs_rel.append(R_rel)
        Cs_rel.append(C_rel)
        
    Rs_rel = np.array(Rs_rel)
    Cs_rel = np.array(Cs_rel)
    
    # 5. Perform 2-second alignment (rotate K-frame to positive Z axis)
    K = min(int(round(2.0 * args.fps)), num_frames - 1)
    d = Cs_rel[K] # relative position at frame K
    
    print(f"[*] 2-second alignment vector (frame {K}): {d}")
    R_align = get_alignment_rotation(d)
    
    final_output = []
    for i in range(num_frames):
        # Similarity / rigid transform
        R_final = R_align @ Rs_rel[i] @ R_align.T
        C_final = R_align @ Cs_rel[i]
        
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :3] = R_final
        pose_4x4[:3, 3] = C_final
        
        final_output.append({
            "frame_id": i,
            "pose": pose_4x4.tolist()
        })
        
    # 6. Output Verification
    poses_arr = np.array([pose["pose"] for pose in final_output])
    Cs_final_arr = poses_arr[:, :3, 3]
    Rs_final_arr = poses_arr[:, :3, :3]
    
    v_final = np.diff(Cs_final_arr, axis=0)
    v_unit = v_final / (np.linalg.norm(v_final, axis=1, keepdims=True) + 1e-9)
    cam_z_world_correct = Rs_final_arr[:-1, :, 2]
    alignment = np.mean(np.sum(v_unit * cam_z_world_correct, axis=1))
    
    print(f"[*] Final Quality Check:")
    print(f"    - Mean Alignment (Direction vs Cam-Z): {alignment:.4f} (Goal: >0.99)")
    print(f"    - Total Path Length: {np.sum(np.linalg.norm(v_final, axis=1)):.2f} meters")
    
    # 7. Save final C4D JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(final_output, f)
    print(f"[+] Success! Final Cinema 4D trajectory saved to: {args.output_json}")

if __name__ == "__main__":
    main()
