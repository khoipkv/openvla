import os
import torch
import pynvml
from typing import Dict, Any, Optional

class H100VLAProfiler:
    def __init__(self, log_dir: str = "./profiler_logs", rank: int = 0):
        self.log_dir = log_dir
        self.rank = rank
        os.makedirs(self.log_dir, exist_ok=True)
        
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.rank)
        self.torch_profiler: Optional[torch.profiler.profile] = None

    def start_torch_profiler(self, warmup_steps: int = 3, active_steps: int = 10):
        self.torch_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=warmup_steps,
                active=active_steps,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        )
        self.torch_profiler.start()

    def step(self):
        if self.torch_profiler:
            self.torch_profiler.step()

    def stop(self):
        if self.torch_profiler:
            self.torch_profiler.stop()

    def sample_nvml_metrics(self) -> Dict[str, Any]:
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        power_usage = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        clock_sm = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
        
        return {
            "gpu_util_pct": utilization.gpu,
            "memory_util_pct": utilization.memory,
            "vram_used_gb": mem_info.used / (1024 ** 3),
            "vram_total_gb": mem_info.total / (1024 ** 3),
            "power_draw_w": power_usage,
            "sm_clock_mhz": clock_sm,
        }

    def close(self):
        pynvml.nvmlShutdown()
