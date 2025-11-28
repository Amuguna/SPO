#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify vLLM process cleanup works correctly with vLLM 0.6+ (spawn multiprocessing)
"""

import subprocess
import time
import psutil
import os


def get_vllm_processes():
    """Find all vLLM related processes"""
    vllm_patterns = [
        'vllm.entrypoints',
        'vllm.worker',
        'vllm.engine',
        'from multiprocessing.spawn',
    ]
    
    vllm_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            cmdline_str = ' '.join(cmdline) if cmdline else ''
            if any(pattern in cmdline_str for pattern in vllm_patterns):
                vllm_procs.append({
                    'pid': proc.pid,
                    'name': proc.name(),
                    'cmdline': cmdline_str[:100]
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return vllm_procs


def get_gpu_memory():
    """Get GPU memory usage"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        for line in lines:
            parts = line.split(',')
            gpu_info.append({
                'index': int(parts[0].strip()),
                'used_mb': int(parts[1].strip()),
                'total_mb': int(parts[2].strip())
            })
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return []


def kill_process_tree(pid: int):
    """
    Kill a process and all its children recursively.
    This is the same logic as in vllm_server.py
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        print(f"  Found {len(children)} child processes")
        
        # Kill children first
        for child in children:
            try:
                print(f"  Killing child process {child.pid} ({child.name()})")
                child.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                print(f"  Error killing child process {child.pid}: {e}")
        
        # Then kill the parent
        try:
            print(f"  Killing parent process {pid}")
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        
        # Wait for all processes to terminate
        gone, alive = psutil.wait_procs(children + [parent], timeout=10)
        print(f"  Terminated: {len(gone)}, Still alive: {len(alive)}")
        
        for p in alive:
            try:
                print(f"  Process {p.pid} still alive, sending SIGKILL")
                p.kill()
            except psutil.NoSuchProcess:
                pass
                
    except psutil.NoSuchProcess:
        print(f"  Process {pid} already terminated")
    except Exception as e:
        print(f"  Error in kill_process_tree: {e}")


def test_vllm_cleanup():
    """Test that vLLM processes can be properly cleaned up"""
    
    print("=" * 60)
    print("vLLM Process Cleanup Test (for vLLM 0.6+ with spawn)")
    print("=" * 60)
    
    # Check initial state
    print("\n[1] Initial State")
    print("-" * 40)
    initial_procs = get_vllm_processes()
    print(f"Existing vLLM processes: {len(initial_procs)}")
    for p in initial_procs:
        print(f"  PID {p['pid']}: {p['cmdline']}")
    
    initial_gpu = get_gpu_memory()
    print(f"\nGPU Memory:")
    for gpu in initial_gpu:
        print(f"  GPU {gpu['index']}: {gpu['used_mb']} / {gpu['total_mb']} MB")
    
    # Start a simple vLLM server on GPU 0
    print("\n[2] Starting vLLM Server")
    print("-" * 40)
    
    # Use a small model for testing if available, otherwise skip actual server test
    test_model = "facebook/opt-125m"  # Small model for testing
    port = 18888
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", test_model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.3",
        "--max-model-len", "512",
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    print(f"Starting vLLM with model: {test_model}")
    print(f"Command: {' '.join(cmd[:5])}...")
    
    try:
        # Start the server
        log_file = open("/tmp/vllm_test.log", "w")
        process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
        print(f"Server started with PID: {process.pid}")
        
        # Wait for server to start and spawn workers
        print("\nWaiting 30 seconds for server to initialize...")
        time.sleep(30)
        
        # Check processes after starting
        print("\n[3] After Starting Server")
        print("-" * 40)
        after_start_procs = get_vllm_processes()
        print(f"vLLM processes: {len(after_start_procs)}")
        for p in after_start_procs:
            print(f"  PID {p['pid']}: {p['cmdline']}")
        
        after_start_gpu = get_gpu_memory()
        print(f"\nGPU Memory:")
        for gpu in after_start_gpu:
            print(f"  GPU {gpu['index']}: {gpu['used_mb']} / {gpu['total_mb']} MB")
        
        # Now test the cleanup
        print("\n[4] Testing Cleanup (kill_process_tree)")
        print("-" * 40)
        kill_process_tree(process.pid)
        
        # Additional cleanup - kill any remaining vLLM processes
        print("\n[5] Additional Cleanup")
        print("-" * 40)
        remaining = get_vllm_processes()
        if remaining:
            print(f"Remaining vLLM processes: {len(remaining)}")
            for p in remaining:
                print(f"  Killing PID {p['pid']}: {p['cmdline']}")
                try:
                    psutil.Process(p['pid']).kill()
                except:
                    pass
        else:
            print("No remaining vLLM processes")
        
        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("CUDA cache cleared")
        except Exception as e:
            print(f"Error clearing CUDA cache: {e}")
        
        time.sleep(5)
        
        # Check final state
        print("\n[6] Final State")
        print("-" * 40)
        final_procs = get_vllm_processes()
        print(f"vLLM processes: {len(final_procs)}")
        for p in final_procs:
            print(f"  PID {p['pid']}: {p['cmdline']}")
        
        final_gpu = get_gpu_memory()
        print(f"\nGPU Memory:")
        for gpu in final_gpu:
            print(f"  GPU {gpu['index']}: {gpu['used_mb']} / {gpu['total_mb']} MB")
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        gpu0_before = after_start_gpu[0]['used_mb'] if after_start_gpu else 0
        gpu0_after = final_gpu[0]['used_mb'] if final_gpu else 0
        memory_released = gpu0_before - gpu0_after
        
        print(f"Processes before cleanup: {len(after_start_procs)}")
        print(f"Processes after cleanup: {len(final_procs)}")
        print(f"GPU 0 memory before: {gpu0_before} MB")
        print(f"GPU 0 memory after: {gpu0_after} MB")
        print(f"Memory released: {memory_released} MB")
        
        if len(final_procs) == 0 and memory_released > 100:
            print("\n✅ SUCCESS: vLLM cleanup works correctly!")
        elif len(final_procs) == 0:
            print("\n⚠️ PARTIAL SUCCESS: Processes cleaned but GPU memory not fully released")
        else:
            print("\n❌ FAILURE: Some vLLM processes still running")
            
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        log_file.close()
        print("\n[Cleanup] Final process cleanup...")
        for pattern in ['vllm.entrypoints', 'vllm.worker']:
            try:
                subprocess.run(["pkill", "-f", "-9", pattern], capture_output=True, timeout=5)
            except:
                pass


if __name__ == "__main__":
    test_vllm_cleanup()
