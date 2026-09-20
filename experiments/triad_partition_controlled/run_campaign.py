#!/usr/bin/env python3
"""Cluster coordinator. --dry-run is read-only and requires no ROCm installation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import tempfile
import time
from topology import capture, check, clean_env, local_cores, COUNTS

HERE = Path(__file__).resolve().parent
CONFIGS = [('malloc', 1), ('hipMalloc', 0), ('hipMalloc', 1), ('hipHostMalloc', 0), ('hipHostMalloc', 1), ('hipMallocManaged', 0), ('hipMallocManaged', 1)]


def write(path, obj):
    """Write obj to path as sorted, indented JSON with a trailing newline."""
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def schedule(devices):
    """Build the predeclared phase list: six traced trials plus one untraced sensitivity round.

    Configuration and device order rotate per repeat, and the concurrent phase
    alternates between first and last, so paired conditions never share a fixed order.
    """
    rows = []
    for repeat in range(1, 8):
        traced = repeat <= 6
        order = CONFIGS[(repeat - 1) % 7:] + CONFIGS[:(repeat - 1) % 7]
        for config_index, (allocator, xnack) in enumerate(order):
            shift = (repeat - 1 + config_index) % len(devices)
            ordered = devices[shift:] + devices[:shift]
            phases = [('isolated', [d]) for d in ordered]
            if repeat % 2:
                phases.append(('concurrent', devices))
            else:
                phases.insert(0, ('concurrent', devices))
            for condition, selected in phases:
                name = condition if condition == 'concurrent' else 'isolated_' + selected[0]['worker']
                rows.append(dict(path=f'{allocator}_x{xnack}/' + (f'trial_{repeat:02d}' if traced else 'sensitivity') + '/' + name,
                    repeat=repeat, allocator=allocator, xnack=xnack, traced=traced, condition=condition, devices=selected))
    return rows


def read_line(conn):
    """Read one newline-terminated JSON message from a worker socket."""
    data = b''
    while len(data) < 16384:
        c = conn.recv(1)
        if not c:
            raise RuntimeError('worker disconnected during readiness')
        if c == b'\n':
            return json.loads(data)
        data += c
    raise RuntimeError('oversized worker message')


def terminate(processes):
    """Send SIGTERM to each process group, then SIGKILL any that do not exit in time."""
    for proc in processes:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 2
    for proc in processes:
        try:
            proc.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

def run_phase(root, spec, binary):
    """Launch all workers for one phase, coordinate the shared window, and validate results."""
    root.mkdir(parents=True, exist_ok=False)
    processes = []
    handles = []
    connections = []
    statuses = {}
    write(root / 'phase.json', spec)
    try:
        with tempfile.TemporaryDirectory(prefix='rq5_') as tmp:
            endpoint = str(Path(tmp) / 's')
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(endpoint)
                server.listen(len(spec['devices']))
                server.settimeout(30)
                for dev in spec['devices']:
                    target = root / dev['worker']
                    target.mkdir()
                    env = clean_env(spec['xnack'])
                    env['ROCR_VISIBLE_DEVICES'] = dev['uuid']
                    command = [str(binary), '--socket', endpoint, '--output', str(target / 'result.json'),
                        '--allocator', spec['allocator'], '--elements', '33554432', '--block-size', '192', '--batch-size', '8',
                        '--expected-bdf', dev['bdf'], '--expected-cus', str(dev['cus']), '--worker-id', dev['worker']]
                    if spec['traced']:
                        command = ['rocprofv3', '--kernel-trace', '--marker-trace', '--output-format', 'csv', '--output-directory', str(target / 'trace'), '--', *command]
                    command = ['numactl', f'--physcpubind={dev["cpu"]}', f'--membind={dev["numa_node"]}', *command]
                    write(target / 'command.json', dict(argv=command, env={k: env[k] for k in ('ROCR_VISIBLE_DEVICES', 'HSA_XNACK')},
                          cleared=['HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL']))
                    log = (target / 'process.log').open('w')
                    handles.append(log)
                    processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
                expected = {d['worker']: d for d in spec['devices']}
                seen = set()
                for _ in spec['devices']:
                    conn, _ = server.accept()
                    connections.append(conn)
                    conn.settimeout(30)
                    ready = read_line(conn)
                    worker = ready['worker']
                    check(worker in expected and worker not in seen, 'unexpected/duplicate ready worker')
                    dev = expected[worker]
                    check(ready['state'] == 'ready' and ready['device']['bdf'] == dev['bdf'] and ready['device']['cus'] == dev['cus'] and ready['affinity'] == [dev['cpu']], 'ready identity/affinity mismatch')
                    seen.add(worker)
                    write(root / worker / 'ready.json', ready)
                lead = time.monotonic_ns() + 100_000_000
                t0 = lead + 200_000_000
                t1 = t0 + 2_000_000_000
                tail = t1 + 200_000_000
                spec.update(lead_ns=lead, t0_ns=t0, t1_ns=t1, tail_ns=tail)
                write(root / 'phase.json', spec)
                for conn in connections:
                    conn.sendall(f'START {lead} {t0} {t1} {tail}\n'.encode())
                # Wait for profiler processes to flush all trace files as well as worker validation.
                deadline = time.monotonic() + 35
                for dev, proc in zip(spec['devices'], processes):
                    statuses[dev['worker']] = proc.wait(timeout=max(.01, deadline - time.monotonic()))
                check(all(v == 0 for v in statuses.values()), 'one or more worker/profiler exits failed')
                for dev in spec['devices']:
                    result = json.loads((root / dev['worker'] / 'result.json').read_text())
                    check(result.get('status') == 'ok' and result.get('validation_errors') == 0, 'worker validation failed')
        write(root / 'status.json', dict(status='ok', workers=statuses))
        return True
    except (Exception, KeyboardInterrupt) as e:
        terminate(processes)
        for dev, proc in zip(spec['devices'], processes):
            statuses[dev['worker']] = proc.returncode
        write(root / 'status.json', dict(status='failed', workers=statuses, reason=str(e)))
        if isinstance(e, KeyboardInterrupt):
            raise
        return False
    finally:
        for conn in connections:
            conn.close()
        for handle in handles:
            handle.close()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=COUNTS, required=True)
    p.add_argument('--binary', type=Path, default=HERE / 'build/triad_worker')
    p.add_argument('--output', type=Path)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not args.execute:
        # Dry run: print the predeclared schedule and parameters without touching hardware.
        devices = [dict(worker=f'device_{i:02d}') for i in range(COUNTS[args.mode])]
        plan = schedule(devices)
        print(json.dumps(dict(mode=args.mode, phase_count=len(plan), traced_pairs_per_config=6, sensitivity_pairs_per_config=1,
            elements_per_array=33554432, arrays_per_process=3, block_size=192, grid_blocks=174763, batch_size=8,
            common_interval_seconds=2, lead_and_tail_seconds=.2, placement='verified first physical APU; one distinct local core per worker',
            slurm_nodes=1, slurm_gpu_resources=2, slurm_time_minutes=20, schedule=plan), indent=2))
        return 0

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    check(os.name == 'posix' and Path('/sys/class/kfd').exists(), 'execution requires an allocated MI300A Linux node')
    check(bool(os.environ.get('SLURM_JOB_ID')), 'execution requires a Slurm allocation')
    root = (args.output or HERE / 'runs' / f'{args.mode}_{os.environ["SLURM_JOB_ID"]}').resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    started = time.monotonic()
    complete = False
    try:
        for cmd in ('rocprofv3', 'numactl', 'amd-smi', 'rocminfo', 'hipconfig'):
            check(shutil.which(cmd) is not None, 'missing ' + cmd)
        check(binary.is_file(), 'missing worker binary: ' + str(binary))
        for name, cmd in [('ldd', ['ldd', str(binary)]), ('profiler_version', ['rocprofv3', '--version'])]:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            (root / (name + '.txt')).write_text(r.stdout + r.stderr)
        devices = capture(root / 'provenance_before', binary, args.mode)
        first = min(d['package_bdf'] for d in devices)
        selected = [d for d in devices if d['package_bdf'] == first]
        check(len({d['numa_node'] for d in selected}) == 1, 'selected package has ambiguous NUMA locality')
        cores = local_cores(selected[0]['numa_node'], len(selected))
        selected = [dict(**d, cpu=cpu, worker=f'device_{i:02d}') for i, (d, cpu) in enumerate(zip(selected, cores))]
        package = selected[0]['package_uuid']
        node = socket.gethostname()
        plan = schedule(selected)
        manifest = dict(schema_version=1, mode=args.mode, node=node, package_uuid=package, devices=selected,
            configurations=CONFIGS, schedule=plan, memory_partition='NPS1', working_set_policy='fixed_per_partition',
            resource_limit_seconds=1200, execution_budget_seconds=1050, source_status='separate variant; GPU validation pending until this campaign passes')
        write(root / 'campaign.json', manifest)
        failures = 0
        executed = 0
        for entry in plan:
            if time.monotonic() - started > 1010:
                break  # reserve cleanup/postflight; never shorten trials
            spec = dict(**entry, mode=args.mode, node=node, package_uuid=package)
            if not run_phase(root / entry['path'], spec, binary):
                failures += 1
            executed += 1
        after = capture(root / 'provenance_after', binary, args.mode)
        check(devices == after, 'device/package identity changed between preflight and postflight')
        complete = executed == len(plan) and failures == 0
        write(root / 'campaign_status.json', dict(status='complete' if complete else 'incomplete', executed_phases=executed, expected_phases=len(plan), failed_phases=failures, elapsed_seconds=time.monotonic() - started))
    except (Exception, KeyboardInterrupt) as e:
        write(root / 'campaign_status.json', dict(status='failed', reason=str(e), elapsed_seconds=time.monotonic() - started))
        raise
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
