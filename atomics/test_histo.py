import pytest
import subprocess
import os

def run(args, xnack=1):
    env = os.environ.copy()
    env["HSA_XNACK"] = str(xnack)
    return subprocess.run(["./histo"] + args, timeout=60, check=True, env=env)

def test_no_cpu_or_gpu():
    with pytest.raises(subprocess.CalledProcessError):
        run([])

def test_1_cpu():
    run(["-c", "1"])

def test_multi_cpu():
    run(["-c", "24"])

def test_1_gpu():
    run(["-g", "1"])

def test_multi_gpu():
    run(["-g", "1024"])

def test_tpb_gpu():
    run(["-g", "1024", "-b", "64"])

def test_1_cpu_gpu():
    run(["-c", "1", "-g", "1"])

def test_multi_cpu_gpu():
    run(["-c", "24", "-g", "1024"])

def test_u64():
    run(["-c", "1", "-g", "1", "-d", "u64"])

def test_f32():
    run(["-c", "1", "-g", "1", "-d", "f32"])

def test_f64():
    run(["-c", "1", "-g", "1", "-d", "f64"])

def test_gpu_no_system():
    run(["-g", "1024", "-S"])

def test_cpu_gpu_no_system():
    run(["-c", "24", "-g", "1024", "-S"])

def test_hip_malloc():
    run(["-c", "1", "-g", "1", "-H"], xnack=0)

def test_relaxed():
    run(["-c", "24", "-g", "1024", "-r"])
