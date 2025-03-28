import sys
import subprocess
import os

def run(args, xnack=1):
    env = os.environ.copy()
    env["HSA_XNACK"] = str(xnack)
    env["HIP_VISBILE_DEVICES"] = "0"

    base_cmd = ["numactl", "-N0", "-m0", "--", "./histo"]
    cmd = base_cmd + args
    print(" ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, timeout=60, check=True, env=env, capture_output=True)
    gpu_rate, cpu_rate = map(float, proc.stdout.split())
    return gpu_rate, cpu_rate

def gpu_args(threads):
    a = ["-g", str(threads), "-b"]
    if threads == 1:
        a.append("1")
    else:
        a.append("64")

    return a

if __name__ == "__main__":
    base_args = ["-t", "0"]
    extra_args = sys.argv[2:]

    op = sys.argv[1]
    if op == "gpu":
        threads_gpu = [1, 64] + list(range(256, 14592+1, 512))
        arglist = list(map(gpu_args, threads_gpu)) 
    elif op == "cpu":
        threads_cpu = list(range(1, 24+1))
        arglist = list(map(lambda c: ["-c", str(c)], threads_cpu)) 
    elif op == "hybrid":
        threads_gpu = list(range(256, 14592+1, 1024))
        threads_cpu = list(range(2, 24+1, 2))
        arglist = []
        for c in threads_cpu:
            for g in threads_gpu:
                arglist.append(["-b", "64", "-g", str(g), "-c", str(c)])
    else:
        raise

    gpu_res = []
    cpu_res = []

    for args in arglist:
        best_gr = 0
        best_cr = 0

        for i in range(5):
            gr, cr = run(base_args + args + extra_args)
            if gr+cr > best_gr+best_cr:
                best_gr = gr
                best_cr = cr

        gpu_res.append(best_gr)
        cpu_res.append(best_cr)

    if op == "gpu":
        for r in gpu_res:
            print(f"{r:g}")
    elif op == "cpu":
        for r in cpu_res:
            print(f"{r:g}")
    elif op == "hybrid":
        for res in (gpu_res, cpu_res):
            rowlen = len(threads_gpu)
            for row in range(0, len(res), rowlen):
                print(",".join(f"{r:g}" for r in res[row:row+rowlen]))
            print()

