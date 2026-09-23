# pasrse_trace.py
import json
import sys
from collections import defaultdict

def analyze_trace(trace_path):
    print(f"Loading trace: {trace_path} ...")
    with open(trace_path, "r") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    
    cpu_ops = defaultdict(float)
    gpu_kernels = defaultdict(float)
    total_cpu_time_us = 0.0
    total_gpu_time_us = 0.0

    for ev in events:
        name = ev.get("name", "")
        dur = ev.get("dur", 0.0)  # duration in microseconds (us)
        cat = ev.get("cat", "")

        # Categorize CPU vs CUDA events
        if cat in ["cpu_op", "user_annotation"]:
            cpu_ops[name] += dur
            total_cpu_time_us += dur
        elif cat in ["kernel", "gpu_memcpy"]:
            gpu_kernels[name] += dur
            total_gpu_time_us += dur

    print("\n" + "=" * 60)
    print("TOP 10 CUDA KERNELS / GPU EVENTS (BY TOTAL DURATION)")
    print("=" * 60)
    sorted_gpu = sorted(gpu_kernels.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, dur_us in sorted_gpu:
        ms = dur_us / 1000.0
        pct = (dur_us / total_gpu_time_us * 100) if total_gpu_time_us > 0 else 0
        print(f"{ms:10.2f} ms ({pct:5.1f}%) | {name[:65]}")

    print("\n" + "=" * 60)
    print("TOP 10 CPU OPERATIONS / SYNCS (BY TOTAL DURATION)")
    print("=" * 60)
    sorted_cpu = sorted(cpu_ops.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, dur_us in sorted_cpu:
        ms = dur_us / 1000.0
        pct = (dur_us / total_cpu_time_us * 100) if total_cpu_time_us > 0 else 0
        print(f"{ms:10.2f} ms ({pct:5.1f}%) | {name[:65]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_trace.py <path_to_trace.json>")
        sys.exit(1)
    analyze_trace(sys.argv[1])
    # Load model
