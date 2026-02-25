# save as clear_gpu.py
import subprocess
import argparse
import os
import signal
import time

def get_gpu_pids(gpu_id: int):
    """
    Get all process IDs (PIDs) running on a specific GPU.
    """
    try:
        # Command to get PIDs, process names, and memory usage for a specific GPU
        cmd = [
            'nvidia-smi',
            '--query-compute-apps=pid,process_name,used_memory',
            f'--id={gpu_id}',
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        pids = []
        output = result.stdout.strip()
        if output:
            for line in output.split('\n'):
                pid, name, memory = line.split(', ')
                pids.append(int(pid))
                print(f"  - Found process PID: {pid}, Name: {name}, Memory: {memory} MiB")
        return pids
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Error: `nvidia-smi` command not found or failed. Is NVIDIA driver installed?")
        return []

def kill_process(pid: int):
    """
    Tries to kill a process first with SIGTERM, then with SIGKILL.
    """
    try:
        # 1. Try a gentle kill (SIGTERM)
        print(f"  - Sending SIGTERM to PID {pid}...")
        os.kill(pid, signal.SIGTERM)
        
        # Wait a moment to see if it terminates
        time.sleep(2)
        
        # 2. Check if the process is still alive and force kill if needed (SIGKILL)
        # os.kill(pid, 0) will raise a ProcessLookupError if the pid is not found.
        os.kill(pid, 0) 
        print(f"  - PID {pid} did not terminate. Sending SIGKILL...")
        os.kill(pid, signal.SIGKILL)
        print(f"  - Successfully killed PID {pid}.")

    except ProcessLookupError:
        print(f"  - Successfully killed PID {pid}.")
    except PermissionError:
        print(f"  - Permission denied to kill PID {pid}. Try running the script with 'sudo'.")
    except Exception as e:
        print(f"  - Error killing process {pid}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Find and kill all processes on specified NVIDIA GPU(s).")
    parser.add_argument('gpu_ids', nargs='+', type=int, help="One or more GPU IDs to clear (e.g., 0 1 3).")
    
    args = parser.parse_args()

    for gpu_id in args.gpu_ids:
        print(f"\nChecking GPU {gpu_id} for running processes...")
        pids = get_gpu_pids(gpu_id)
        
        if not pids:
            print(f"GPU {gpu_id} has no running compute processes.")
            continue
        
        print(f"Attempting to kill {len(pids)} process(es) on GPU {gpu_id}...")
        for pid in pids:
            kill_process(pid)

if __name__ == "__main__":
    main()