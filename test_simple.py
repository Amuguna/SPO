#!/usr/bin/env python
"""Simple test for process tree killing"""

import subprocess
import time
import psutil
import os
import sys

print("=" * 50, flush=True)
print("Simple vLLM Cleanup Test", flush=True)
print("=" * 50, flush=True)

def get_gpu_memory():
    """Get GPU memory usage"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return [int(x.split(',')[1].strip()) for x in result.stdout.strip().split('\n')]
    except:
        return []

def get_vllm_procs():
    """Get vLLM processes"""
    patterns = ['vllm.entrypoints', 'vllm.worker', 'vllm.engine', 'multiprocessing.spawn']
    procs = []
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(proc.info.get('cmdline') or [])
            if any(p in cmd for p in patterns):
                procs.append((proc.pid, cmd[:80]))
        except:
            pass
    return procs

def kill_tree(pid):
    """Kill process tree"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        print(f"  Parent PID: {pid}, Children: {len(children)}", flush=True)
        
        for c in children:
            try:
                print(f"  Killing child {c.pid}", flush=True)
                c.kill()
            except:
                pass
        
        try:
            parent.kill()
        except:
            pass
        
        psutil.wait_procs(children + [parent], timeout=10)
        print("  Process tree killed", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)

# Step 1: Check initial state
print("\n[1] Initial GPU Memory:", flush=True)
mem = get_gpu_memory()
for i, m in enumerate(mem):
    print(f"  GPU {i}: {m} MB", flush=True)

print("\n[2] Current vLLM processes:", flush=True)
procs = get_vllm_procs()
print(f"  Found {len(procs)} processes", flush=True)
for pid, cmd in procs:
    print(f"  {pid}: {cmd}", flush=True)

# Step 2: Start vLLM server
print("\n[3] Starting vLLM server...", flush=True)

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", "facebook/opt-125m",
    "--port", "19999",
    "--dtype", "float16",
    "--gpu-memory-utilization", "0.2",
    "--max-model-len", "256",
]

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

try:
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  Started with PID: {proc.pid}", flush=True)
    
    print("  Waiting 20s for initialization...", flush=True)
    time.sleep(20)
    
    print("\n[4] After start - vLLM processes:", flush=True)
    procs = get_vllm_procs()
    print(f"  Found {len(procs)} processes", flush=True)
    for pid, cmd in procs:
        print(f"  {pid}: {cmd}", flush=True)
    
    print("\n[5] GPU Memory after start:", flush=True)
    mem_after = get_gpu_memory()
    for i, m in enumerate(mem_after):
        print(f"  GPU {i}: {m} MB", flush=True)
    
    # Step 3: Kill using process tree
    print("\n[6] Killing process tree...", flush=True)
    kill_tree(proc.pid)
    
    # Additional cleanup
    print("\n[7] Additional cleanup...", flush=True)
    for p in get_vllm_procs():
        try:
            print(f"  Killing {p[0]}", flush=True)
            psutil.Process(p[0]).kill()
        except:
            pass
    
    # Clear CUDA
    try:
        import torch
        torch.cuda.empty_cache()
        print("  CUDA cache cleared", flush=True)
    except:
        pass
    
    time.sleep(5)
    
    print("\n[8] Final state - vLLM processes:", flush=True)
    final_procs = get_vllm_procs()
    print(f"  Found {len(final_procs)} processes", flush=True)
    
    print("\n[9] Final GPU Memory:", flush=True)
    mem_final = get_gpu_memory()
    for i, m in enumerate(mem_final):
        print(f"  GPU {i}: {m} MB", flush=True)
    
    # Summary
    print("\n" + "=" * 50, flush=True)
    print("RESULT:", flush=True)
    if len(final_procs) == 0:
        print("✅ All vLLM processes terminated!", flush=True)
    else:
        print("❌ Some processes still running", flush=True)
    
    if mem_after and mem_final:
        released = mem_after[0] - mem_final[0]
        print(f"GPU 0 memory released: {released} MB", flush=True)
        if released > 50:
            print("✅ GPU memory released!", flush=True)
        else:
            print("⚠️ GPU memory may not be fully released", flush=True)
    
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()

finally:
    # Final cleanup
    subprocess.run(["pkill", "-f", "-9", "vllm.entrypoints"], capture_output=True)
    subprocess.run(["pkill", "-f", "-9", "opt-125m"], capture_output=True)
    print("\nTest complete.", flush=True)
