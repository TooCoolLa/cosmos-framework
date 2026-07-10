# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.inference.common.init import init_script, is_rank0
init_script()

import os
import json
import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any

import pydantic
import torch
import tyro
import numpy as np
import torchvision
from PIL import Image

# 优先导入 log
from cosmos_framework.utils import log

# ==================== 运行时热补丁 (Monkey Patch) ====================
def my_read_media_frames(path: Path, max_frames: int) -> tuple[torch.Tensor, float]:
    """Read an image or video into a uint8 tensor of shape (C, T, H, W).
    Enhanced version to support image directories and wildcard glob pattern.
    """
    is_pattern = "*" in path.name or "?" in path.name
    if path.is_dir() or is_pattern:
        if is_pattern:
            img_paths = sorted(list(path.parent.glob(path.name)))
        else:
            img_paths = sorted([
                p for p in path.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ])
            
        if not img_paths:
            raise ValueError(f"No matching images found for path: {path}")
            
        img_paths = img_paths[:max_frames]
        frames_list = []
        ref_size = None
        for img_p in img_paths:
            with img_p.open("rb") as f:
                img = Image.open(f).convert("RGB")
            if ref_size is None:
                ref_size = img.size
            elif img.size != ref_size:
                img = img.resize(ref_size, Image.LANCZOS)
            img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1)
            frames_list.append(img_t)
        frames = torch.stack(frames_list, dim=1)
        return frames, 24.0

    ext = path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        with path.open("rb") as f:
            image = Image.open(f).convert("RGB")
        frames = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(1)
        return frames, 1.0
        
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise ValueError(f"Unsupported media extension: {ext}")
        
    frames, _, info = torchvision.io.read_video(str(path), pts_unit="sec")
    frames = frames[:max_frames].permute(0, 3, 1, 2).permute(1, 0, 2, 3)
    fps = float(info.get("video_fps", 24.0))
    return frames, fps

# 执行热补丁注入，劫持底层的 read_media_frames
import cosmos_framework.inference.vision as vision_module
vision_module.read_media_frames = my_read_media_frames
log.info("Successfully monkey-patched read_media_frames to support directory frame list loading.")
# ====================================================================

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import SampleOutputs, SetupOverrides, tyro_cli
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.utils import log
from cosmos_framework.scripts.inference import InferenceArgs

from cosmos_framework.inference.action import get_action_sample_data
from cosmos_framework.inference.inference import _finalize_data_batch
from cosmos_framework.tools.visualize.video import save_img_or_video


def inference(args: InferenceArgs):
    from cosmos_framework.inference.common.inference import sync_distributed_errors

    with sync_distributed_errors():
        if args.setup.output_dir is None:
            raise ValueError("'output_dir' is required")
        setup_args = args.setup.build_setup()
        init_output_dir(setup_args.output_dir)
        log.debug(f"{args.__class__.__name__}({args})")
        
        sample_overrides_list = setup_args.get_sample_overrides_cls().from_files(
            args.input_files, overrides=setup_args.sample_overrides
        )
        log.info(f"Loaded {len(sample_overrides_list)} samples for multi-threaded inference")
        
        # 实例化 pipe，载入模型与权重 (这一步耗时较长)
        pipe = setup_args.get_inference_cls().create(setup_args)
        model = pipe.model
        device = model.tensor_kwargs["device"]
        
        sample_args_list = []
        for overrides in sample_overrides_list:
            assert overrides.name
            
            # 【断点续推防覆写】：若该 Batch 对应的目标 JSON 文件已算好存在，直接跳过
            result_file = setup_args.output_dir / f"{overrides.name}.json"
            if result_file.exists():
                continue
            
            # 1. 优先将 output_dir 赋给 overrides 从而在 build_sample 中顺利通过校验
            overrides.output_dir = setup_args.output_dir / overrides.name
            
            # 2. 构建已通过全部强校验的 sample args (保留第一帧图片的路径以通过后缀合法性校验)
            sa = overrides.build_sample(model_config=pipe.model_config)
            sample_args_list.append(sa)

    # 1. 初始化限制容量为 4 的反压同步队列
    load_queue = queue.Queue(maxsize=4)
    save_queue = queue.Queue(maxsize=4)

    # 2. 定义加载线程工作逻辑 (CPU 读视频 + Resize)
    def loader_worker():
        log.info("Loader thread started.")
        try:
            for sa in sample_args_list:
                try:
                    # 动态把第一帧图片文件的路径替换为父目录作为文件夹图片序列读入
                    vpath = sa.vision_path
                    if vpath:
                        vp = Path(vpath)
                        if vp.is_file() and vp.name.startswith("frame_"):
                            vpath = str(vp.parent)
                            
                    # 预先在 CPU 上读取视频并打包以防抢占 GPU 锁
                    data_batch = get_action_sample_data(
                        model.config,
                        batch_size=1,
                        prompt=sa.prompt,
                        vision_path=vpath,  # 传入真实的图片文件夹目录以读入全部帧 (由 monkeypatch 读入)
                        model_mode=sa.model_mode,
                        action_path=sa.action_path,
                        domain_name=sa.domain_name,
                        view_point=sa.view_point,
                        resolution=sa.resolution,
                        action_chunk_size=sa.action_chunk_size,
                        max_action_dim=model.config.max_action_dim,
                        fps=sa.fps,
                        device="cpu"
                    )
                    load_queue.put((sa, data_batch))
                except Exception as e:
                    log.error(f"Loader failed on {sa.name}: {e}")
                    load_queue.put((sa, e))
        finally:
            # 兜底：无论以任何形式退出/崩溃，百分之百保证压入 None 哨兵，防止推理线程挂起
            load_queue.put(None)
            log.info("Loader thread completed.")

    # 3. 定义推理线程工作逻辑 (GPU Forward + Action Extraction, Skip VAE Decode)
    def inference_worker():
        log.info("Inference thread started.")
        try:
            is_distilled = model.config.fixed_step_sampler_config is not None
            sampler = model.fixed_step_sampler if is_distilled else None
            
            # 提取公共参数
            guidance = sample_args_list[0].guidance
            shift = sample_args_list[0].shift
            sigma_max = sample_args_list[0].sigma_max
            num_steps = sample_args_list[0].num_steps
            normalize_cfg = sample_args_list[0].normalize_cfg

            while True:
                item = load_queue.get()
                if item is None:
                    break
                sa, data_batch = item
                if isinstance(data_batch, Exception):
                    save_queue.put((sa, data_batch))
                    continue
                    
                data_batch_gpu = None
                outputs = None
                
                try:
                    # 传到 GPU
                    data_batch_gpu = _finalize_data_batch(data_batch, batch_size=1, model=model)
                    for k, v in data_batch_gpu.items():
                        if isinstance(v, torch.Tensor):
                            data_batch_gpu[k] = v.to(device=device)
                        elif isinstance(v, list) and isinstance(v[0], list) and isinstance(v[0][0], torch.Tensor):
                            data_batch_gpu[k] = [[v[0][0].to(device=device)]]
                    
                    # 运行生成 (去噪采样生成 vision_latent 与 action)
                    outputs = model.generate_samples_from_batch(
                        data_batch_gpu,
                        sampler=sampler,
                        guidance=guidance,
                        seed=[sa.seed],
                        num_steps=num_steps,
                        align_num_steps=num_steps,
                        shift=shift,
                        sigma_max=sigma_max,
                        normalize_cfg=normalize_cfg,
                        n_sample=1
                    )
                    
                    # 【核心提速】：完全不进行耗时庞大的 VAE 解码，只将预测的 action 位姿数据带回 CPU
                    output = {}
                    if "action" in outputs:
                        output["action"] = outputs["action"][0].squeeze(0).cpu()
                    if "sound" in outputs:
                        # 如无 sound_gen 默认不会触发，如有则保留
                        sound = model.decode_sound(outputs.pop("sound")[0])
                        output["sound"] = sound.cpu()
                        
                    save_queue.put((sa, output))
                except Exception as e:
                    log.error(f"Inference failed on {sa.name}: {e}")
                    save_queue.put((sa, e))
                finally:
                    # 显式销毁 GPU 临时变量
                    del data_batch_gpu, outputs
                    torch.cuda.empty_cache()  # 辅助回收缓存区
        finally:
            # 兜底：保证向保存线程池发送退出信号，绝不让主线程死等
            save_queue.put(None)
            log.info("Inference thread completed.")

    # 4. 定义保存线程逻辑 (只写 JSON 动作文件，不保存任何 mp4 视频文件)
    def save_task(sa, output):
        # 结果文件名：batch_{batchindex}_{start_frame}_{end_frame}.json
        result_file = setup_args.output_dir / f"{sa.name}.json"
        
        if isinstance(output, Exception):
            # 保存错误信息
            msg = f"Error generating sample '{sa.name}': {output}"
            err_output = SampleOutputs(
                args=sa.model_dump(mode="json"), status="error", message=msg, stack_trace=""
            )
            result_file.write_text(err_output.model_dump_json())
            return
            
        try:
            content = {}
            if "action" in output:
                content["action"] = output["action"].tolist()
                
            sample_output = SampleOutputs(
                args=sa.model_dump(mode="json"),
                status="success",
                outputs=[{"content": content, "files": []}]
            )
            result_file.write_text(sample_output.model_dump_json())
            log.success(f"Saved {result_file.name} successfully")
        except Exception as e:
            log.error(f"Saver failed on {sa.name}: {e}")

    # 启动 Loader & Inference 线程
    t_loader = threading.Thread(target=loader_worker, name="LoaderThread")
    t_infer = threading.Thread(target=inference_worker, name="InferThread")
    
    t_loader.start()
    t_infer.start()

    # 用 ThreadPoolExecutor 运行 4 个并行保存线程
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="SaverPool") as executor:
        while True:
            item = save_queue.get()
            if item is None:
                break
            sa, output = item
            executor.submit(save_task, sa, output)
            
    t_loader.join()
    t_infer.join()
    log.success("All pipeline batches processed and saved.")


def main():
    args = tyro_cli(
        InferenceArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    inference(args)


if __name__ == "__main__":
    main()
