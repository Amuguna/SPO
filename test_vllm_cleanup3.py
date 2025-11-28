#!/usr/bin/env python
"""
Test script to verify vLLM process cleanup works correctly for vLLM 0.6+ (spawn mode)
Uses the correct Python environment with vLLM installed.
"""
import subprocess
import time
import os
import sys

# Use the correct Python with vLLM
VLLM_PYTHON = "/venv/vineppo/bin/python"

def get_gpu_memory():
    """Get GPU memory usage"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        return [(int(x.split(',')[0].strip()), int(x.split(',')[1].strip())) for x in lines]
    except:
        return []

def get_vllm_processes():
    """Get all vLLM related processes"""
    try:
        result = subprocess.run(
            ["pgrep", "-af", "vllm"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return [line for line in result.stdout.strip().split('\n') if line]
        return []
    except:
        return []

def get_all_child_processes(pid):
    """Get all child processes using ps command"""
    try:
        # Get all descendants
        result = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True, text=True
        )
        children = []
        if result.returncode == 0:
            for child_pid in result.stdout.strip().split('\n'):
                if child_pid:
                    children.append(int(child_pid))
                    # Recursively get grandchildren
                    children.extend(get_all_child_processes(int(child_pid)))
        return children
    except:
        return []

def kill_process_tree_manual(pid):
    """Kill process tree using manual method (without psutil)"""
    print(f"  Parent PID: {pid}")
    
    # Get all children recursively
    children = get_all_child_processes(pid)
    print(f"  Found {len(children)} child processes: {children}")
    
    # Kill children first (in reverse order - deepest first)
    for child_pid in reversed(children):
        try:
            print(f"  Killing child {child_pid}")
            os.kill(child_pid, 9)  # SIGKILL
        except ProcessLookupError:
            print(f"  Child {child_pid} already terminated")
        except Exception as e:
            print(f"  Error killing {child_pid}: {e}")
    
    # Kill parent
    try:
        print(f"  Killing parent {pid}")
        os.kill(pid, 9)
    except ProcessLookupError:
        print(f"  Parent {pid} already terminated")
    except Exception as e:
        print(f"  Error killing parent: {e}")

def kill_process_tree_psutil(pid):
    """Kill process tree using psutil"""
    try:
        import psutil
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        print(f"  Parent PID: {pid}")
        print(f"  Found {len(children)} child processes:")
        for child in children:
            try:
                cmdline = ' '.join(child.cmdline()[:3]) if child.cmdline() else 'N/A'
                print(f"    - PID {child.pid}: {child.name()} - {cmdline}")
            except:
                print(f"    - PID {child.pid}: (info unavailable)")
        
        # Kill children first
        for child in children:
            try:
                child.kill()
                print(f"  Killed child {child.pid}")
            except psutil.NoSuchProcess:
                pass
        
        # Kill parent
        try:
            parent.kill()
            print(f"  Killed parent {pid}")
        except psutil.NoSuchProcess:
            pass
        
        # Wait for termination
        gone, alive = psutil.wait_procs(children + [parent], timeout=10)
        print(f"  Terminated: {len(gone)}, Still alive: {len(alive)}")
        
        for p in alive:
            try:
                p.kill()
            except:
                pass
                
    except Exception as e:
        print(f"  psutil error: {e}, falling back to manual method")
        kill_process_tree_manual(pid)

def main():
    print("=" * 60)
    print("vLLM Process Cleanup Test (for vLLM 0.6+ with spawn)")
    print("=" * 60)
    
    # Check vLLM version
    print(f"\nUsing Python: {VLLM_PYTHON}")
    result = subprocess.run([VLLM_PYTHON, "-c", "import vllm; print(vllm.__version__)"], 
                          capture_output=True, text=True)
    print(f"vLLM version: {result.stdout.strip()}")
    
    # Step 1: Check initial state
    print("\n[1] Initial State")
    print("-" * 40)
    
    initial_procs = get_vllm_processes()
    print(f"Existing vLLM processes: {len(initial_procs)}")
    for proc in initial_procs:
        print(f"  {proc}")
    
    initial_mem = get_gpu_memory()
    print("\nGPU Memory:")
    for i, (used, total) in enumerate(initial_mem):
        print(f"  GPU {i}: {used} / {total} MB")
    
    # Step 2: Start vLLM server on GPU 0
    print("\n[2] Starting vLLM Server")
    print("-" * 40)
    
    model = "facebook/opt-125m"
    port = 18888
    
    print(f"Starting vLLM with model: {model}")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    cmd = [
        VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", "0.5",
        "--max-model-len", "512"
    ]
    
    print(f"Command: {' '.join(cmd[:6])}...")
    
    log_file = open("/tmp/vllm_test_server.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    
    print(f"Server started with PID: {proc.pid}")
    print("\nWaiting for server to initialize (checking every 5s)...")
    
    # Wait and check periodically
    server_ready = False
    for i in range(12):  # 60 seconds max
        time.sleep(5)
        procs = get_vllm_processes()
        mem = get_gpu_memory()
        gpu0_used = mem[0][0] if mem else 0
        print(f"  {(i+1)*5}s: {len(procs)} vLLM procs, GPU 0: {gpu0_used} MB", end="")
        
        # Check if server is responding
        try:
            import urllib.request
            req = urllib.request.Request(f"http://localhost:{port}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(" - Server responding!")
                    server_ready = True
                    break
        except:
            print()
    
    if not server_ready:
        print("\nServer may not be fully ready, but continuing with test...")
    
    # Wait a bit more to ensure workers are spawned
    time.sleep(5)
    
    # Step 3: Check state after startup
    print("\n[3] After Starting Server")
    print("-" * 40)
    
    procs_after_start = get_vllm_processes()
    print(f"vLLM processes: {len(procs_after_start)}")
    for proc_info in procs_after_start:
        print(f"  {proc_info[:120]}")
    
    mem_after_start = get_gpu_memory()
    print("\nGPU Memory:")
    for i, (used, total) in enumerate(mem_after_start):
        marker = " <-- vLLM" if i == 0 and used > 100 else ""
        print(f"  GPU {i}: {used} / {total} MB{marker}")
    
    # Step 4: Test cleanup
    print("\n[4] Testing Cleanup (kill_process_tree)")
    print("-" * 40)
    
    kill_process_tree_psutil(proc.pid)
    
    print("\nWaiting 5 seconds for memory release...")
    time.sleep(5)
    
    # Step 5: Additional cleanup (simulate what our code does)
    print("\n[5] Additional Cleanup (pkill fallback)")
    print("-" * 40)
    
    patterns = ["vllm.entrypoints", "vllm.worker", "vllm.engine", "multiprocessing.spawn"]
    for pattern in patterns:
        try:
            result = subprocess.run(["pkill", "-f", "-9", pattern], capture_output=True)
            print(f"  pkill '{pattern}': returncode={result.returncode}")
        except Exception as e:
            print(f"  Error: {e}")
    
    time.sleep(3)
    
    # Step 6: Final state
    print("\n[6] Final State")
    print("-" * 40)
    
    final_procs = get_vllm_processes()
    print(f"Remaining vLLM processes: {len(final_procs)}")
    for proc_info in final_procs:
        # Filter out this test script itself
        if "test_vllm" not in proc_info:
            print(f"  {proc_info[:120]}")
    
    final_mem = get_gpu_memory()
    print("\nGPU Memory:")
    for i, (used, total) in enumerate(final_mem):
        print(f"  GPU {i}: {used} / {total} MB")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    gpu0_before = mem_after_start[0][0] if mem_after_start else 0
    gpu0_after = final_mem[0][0] if final_mem else 0
    
    # Filter out test script from process count
    actual_vllm_procs_before = len([p for p in procs_after_start if "test_vllm" not in p])
    actual_vllm_procs_after = len([p for p in final_procs if "test_vllm" not in p])
    
    print(f"GPU 0 memory before cleanup: {gpu0_before} MB")
    print(f"GPU 0 memory after cleanup:  {gpu0_after} MB")
    print(f"Memory released: {gpu0_before - gpu0_after} MB")
    print(f"vLLM processes cleaned: {actual_vllm_procs_before} -> {actual_vllm_procs_after}")
    
    if gpu0_after < 500 and actual_vllm_procs_after == 0:
        print("\n✅ SUCCESS: vLLM cleanup works correctly!")
    elif actual_vllm_procs_after == 0:
        print("\n⚠️ PARTIAL: Processes cleaned, GPU memory may take time to release")
    else:
        print("\n❌ FAILURE: vLLM processes or GPU memory not fully released")
        
    log_file.close()
    
    # Show server log if there were issues
    if gpu0_before < 100:
        print("\n[DEBUG] Server log (last 30 lines):")
        print("-" * 40)
        try:
            with open("/tmp/vllm_test_server.log", "r") as f:
                lines = f.readlines()
                for line in lines[-30:]:
                    print(line.rstrip())
        except:
            pass

if __name__ == "__main__":
    main()
