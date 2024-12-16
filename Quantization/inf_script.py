import subprocess
import re

def run_inference_script(num_runs=50):
    inference_times = []

    for i in range(num_runs):
        process = subprocess.Popen(
            ["python", "quantization_ucf_infer.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        for line in stdout.splitlines():
            if "average inference time (ms):" in line:
                match = re.search(r"average inference time \(ms\):\s+([\d.]+)", line)
                if match:
                    time_val = float(match.group(1))
                    inference_times.append(time_val)
                    break

    if inference_times:
        overall_average = sum(inference_times) / len(inference_times)
        print(f"Collected {len(inference_times)} inference times.")
        print(f"Per-run average inference times: {inference_times}")
        print(f"Overall average inference time (ms): {overall_average}")
    else:
        print("No inference times were collected.")

if __name__ == "__main__":
    run_inference_script(50)
