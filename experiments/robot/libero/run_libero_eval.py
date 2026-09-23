"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import torch
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# Custom Profiler Module
from vla_hardware_profiler import H100VLAProfiler  # <--- [PROFILER MOD 1]


@dataclass
class GenerateConfig:
    # Model-specific parameters
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    # LIBERO environment-specific parameters
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    # Utils
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    seed: int = 7


class PhaseSweepAnalyzer:
    def __init__(self, log_filename="phase_sweep_results.log"):
        self.temporal_metrics = []
        self.structural_metrics = {}
        self.log_filename = log_filename

    def log_temporal(self, duration_ms):
        self.temporal_metrics.append(duration_ms)

    def print_results(self):
        output_lines = []
        output_lines.append("\n" + "="*70)
        output_lines.append(" PHASE-ADAPTIVE SWEEP RESULTS: PEAK-TO-MEAN RATIO ANALYSIS ")
        output_lines.append("="*70)
        
        # 1. Temporal Phase Results
        if self.temporal_metrics:
            t_vals = np.array(self.temporal_metrics)
            t_peak = np.max(t_vals)
            t_mean = np.mean(t_vals)
            t_p2m = t_peak / t_mean if t_mean > 0 else 1.0
            output_lines.append(
                f"[Temporal Windows]   Peak: {t_peak:6.2f}ms | Mean: {t_mean:6.2f}ms | Peak-to-Mean: {t_p2m:4.2f}x"
            )

        # 2. Structural Hook Results
        if self.structural_metrics:
            output_lines.append("-" * 70)
            output_lines.append(f"{'Layer Name':<40} | {'Peak Entropy':<12} | {'Mean Entropy':<12} | {'P-to-M Ratio':<10}")
            output_lines.append("-" * 70)
            for name, entropies in self.structural_metrics.items():
                s_vals = np.array(entropies)
                s_peak = np.max(s_vals)
                s_mean = np.mean(s_vals)
                s_p2m = s_peak / s_mean if s_mean > 0 else 1.0
                output_lines.append(f"{name[:40]:<40} | {s_peak:12.2f} | {s_mean:12.2f} | {s_p2m:9.2f}x")
        output_lines.append("="*70 + "\n")

        # Join output
        report = "\n".join(output_lines)

        # Print to stdout
        print(report)

        # Save to log file
        with open(self.log_filename, "a") as f:
            f.write(report)
        print(f"[PhaseSweepAnalyzer] Results logged to {os.path.abspath(self.log_filename)}")

sweep_analyzer = PhaseSweepAnalyzer()

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    def make_entropy_hook(layer_name):
        def hook(module, input, output):
            tensor = output[0] if isinstance(output, tuple) else output
        
            if isinstance(tensor, torch.Tensor):
                with torch.no_grad():
                    t = tensor.detach().float()
                    act_norm = torch.norm(t, dim=1)
                    if act_norm.sum() < 1e-5:
                        return

                    probs = torch.softmax(act_norm, dim=0)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-7)).mean()
                    if layer_name not in sweep_analyzer.structural_metrics:
                        sweep_analyzer.structural_metrics[layer_name] = []
                    sweep_analyzer.structural_metrics[layer_name].append(entropy)
        return hook

    for name, module in model.named_modules():
        if "layers" in name or "block" in name or "backbone" in name:
            if name.endswith("0") or name.endswith("layer"): # Avoid over-hooking every single submodule
                module.register_forward_hook(make_entropy_hook(name))			

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # <--- [PROFILER MOD 2]: Instantiate Profiler ---
    profiler_log_dir = os.path.join(cfg.local_log_dir, "h100_profile")
    profiler = H100VLAProfiler(log_dir=profiler_log_dir)
    profiler.start_torch_profiler(warmup_steps=3, active_steps=10)
    # -----------------------------------------------

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # <--- [PROFILER MOD 3]: Wrap Step & Measure Bottlenecks ---
                    t0 = time.perf_counter()

                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        profiler.step()
                        continue

                    # 1. Preprocessing (CPU)
                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    t1 = time.perf_counter()

                    # 2. VLA Inference (H100 GPU)
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )
                    action = normalize_gripper_action(action, binarize=True)
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)
                    
                    # Force stream sync so CPU timer reflects exact CUDA runtime
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()

                    # 3. Environment Step (MuJoCo / EGL)
                    obs, reward, done, info = env.step(action.tolist())
                    torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    
                    step_duration_ms = (time.perf_counter() - t1) * 1000.0

                    sweep_analyzer.log_temporal(step_duration_ms)

                    # Hardware Metrics Log
                    hw_stats = profiler.sample_nvml_metrics()
                    if t % 20 == 0:
                        log_line = (
                            f"[Step {t:03d}] Preproc: {(t1 - t0)*1000:.2f}ms | "
                            f"VLA Forward: {(t2 - t1)*1000:.2f}ms | "
                            f"MuJoCo Step: {(t3 - t2)*1000:.2f}ms | "
                            f"VRAM: {hw_stats['vram_used_gb']:.1f}/{hw_stats['vram_total_gb']:.1f} GB | "
                            f"Power: {hw_stats['power_draw_w']:.0f}W"
                        )
                        print(log_line)
                        log_file.write(log_line + "\n")

                    profiler.step()
                    # ------------------------------------------------------------

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # <--- [PROFILER MOD 4]: Flush and Clean Up ---
    sweep_analyzer.print_results()
    profiler.stop()
    profiler.close()
    print(f"H100 Profiler traces saved to: {profiler_log_dir}")
    # -----------------------------------------------

    log_file.close()

    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
