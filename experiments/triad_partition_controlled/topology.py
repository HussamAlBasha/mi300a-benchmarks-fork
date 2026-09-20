"""Fail-closed physical-package mapping. No ordinal-to-package arithmetic."""
from __future__ import annotations
from pathlib import Path
import json
import os
import re
import subprocess

COUNTS = {'spx': 1, 'tpx': 3, 'cpx': 6}


def check(ok, why):
    """Raise ValueError(why) unless ok is truthy (fail-closed guard)."""
    if not ok:
        raise ValueError(why)


def canonical_bdf(value):
    """Normalize a PCI bus address to canonical domain:bus:device.function form."""
    m = re.fullmatch(r'([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value.strip())
    check(m is not None, 'invalid PCI BDF: ' + value)
    d, b, s, f = (int(x, 16) for x in m.groups())
    return f'{d:04x}:{b:02x}:{s:02x}.{f:x}'


def location(bdf):
    """Return (domain, packed location_id) for a PCI BDF, matching KFD's encoding."""
    d, b, s, f = (int(x, 16) for x in re.split('[:.]', canonical_bdf(bdf)))
    return d, (b << 8) | (s << 3) | f

def render_package(props, mode, smi, sysfs=Path('/sys')):
    """Resolve PCI and XCP render devices using KFD's hardware identifiers.

    KFD exports PCI location OR partition node_id for partitioned GPUs.
    With a physical function-zero MI300A and at most six partitions, the
    partition occupies the low three bits. The amdgpu_xcp_N platform-device
    number is unrelated to package membership and is deliberately ignored.
    See Linux v6.12 amdkfd/kfd_topology.c and amdxcp/amdgpu_xcp_drv.c.
    """
    minor = int(props['drm_render_minor'])
    render = sysfs / f'class/drm/renderD{minor}/device'
    dev = render.resolve(strict=True)
    domain = int(props.get('domain', 0))
    loc = int(props['location_id'])
    count = COUNTS[mode]
    candidates = []
    for package in smi:
        pdomain, base = location(package['bdf'])
        if domain != pdomain:
            continue
        if count == 1:
            matches = loc == base
        else:
            # Reject unsupported physical function encodings, rather than
            # rounding an arbitrary logical address to a guessed package.
            matches = (base & 7) == 0 and (loc & ~7) == base and (loc & 7) < count
        if matches:
            candidates.append(package)
    check(len(candidates) == 1,
          f'ambiguous or missing KFD/AMD SMI package for renderD{minor}: '
          f'domain={domain}, location_id={loc}, sysfs={dev}')
    physical = canonical_bdf(candidates[0]['bdf'])
    pci = (sysfs / 'bus/pci/devices' / physical).resolve(strict=True)
    if re.fullmatch(r'amdgpu_xcp_\d+', dev.name):
        check(count > 1, 'XCP platform device unexpected in SPX')
        evidence = 'kfd_location_partition_bits+amd_smi_pci'
    else:
        direct = (dev / 'physfn').resolve(strict=True) if (dev / 'physfn').exists() else dev
        check(canonical_bdf(direct.name) == physical and direct == pci,
              f'render-device PCI identity disagrees with KFD/AMD SMI: {dev}')
        evidence = 'render_pci+kfd_location+amd_smi'
    numa = int((pci / 'numa_node').read_text())
    check(numa >= 0, f'missing NUMA locality for physical package {physical}')
    return dict(physical_bdf=physical, numa_node=numa, sysfs_device=str(dev),
                sysfs_physical_device=str(pci), package_mapping=evidence,
                partition_id=(loc & 7) if count > 1 else 0)

def rocr_agents(text):
    """Parse rocminfo output into a list of GPU agent identity records."""
    result = []
    for block in re.split(r'(?m)^Agent\s+\d+\s*$', text):
        if not re.search(r'Device Type:\s+GPU', block):
            continue

        def field(key):
            m = re.search(r'^\s*' + re.escape(key) + r':\s+(\S+)', block, re.M)
            check(m is not None, 'missing ROCr ' + key)
            return m[1]

        result.append(dict(
            location_id=int(field('BDFID')),
            uuid=field('Uuid'),
            cus=int(field('Compute Unit')),
            xnack='xnack+' if 'xnack+' in block else 'xnack-' if 'xnack-' in block else 'unknown',
        ))
    return result


def smi_devices(data):
    """Extract the two distinct physical APU (bdf, uuid) pairs from AMD SMI JSON."""
    found = []

    def walk(obj):
        if isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, dict):
            lower = {str(k).lower(): v for k, v in obj.items()}
            if 'bdf' in lower and 'uuid' in lower:
                found.append(dict(bdf=canonical_bdf(str(lower['bdf'])), uuid=str(lower['uuid'])))
            else:
                for v in obj.values():
                    walk(v)

    walk(data)
    check(len(found) == 2 and len({r['bdf'] for r in found}) == 2 and len({r['uuid'] for r in found}) == 2,
          'AMD SMI must identify two distinct physical APUs')
    return found

def map_devices(mode, hip, rocr, kfd, smi):
    """Cross-check HIP, ROCr, KFD and AMD SMI identities into one device list per package."""
    count = COUNTS[mode]
    check(len(hip) == len(rocr) == len(kfd) == 2 * count, 'node-wide device counts differ')
    check(len({h['bdf'] for h in hip}) == len(hip), 'duplicate HIP BDF')
    check(len({k['unique_id'] for k in kfd}) == len(kfd), 'duplicate KFD identity')
    check(len({r['uuid'] for r in rocr}) == len(rocr), 'duplicate ROCr UUID')
    result = []
    for h in hip:
        domain, loc = location(h['bdf'])
        candidates = [k for k in kfd if k['location_id'] == loc and k['domain'] == domain]
        check(len(candidates) == 1, 'ambiguous HIP/KFD mapping')
        k = candidates[0]
        agents = [r for r in rocr if r['location_id'] == loc and int(r['uuid'].removeprefix('GPU-'), 16) == k['unique_id']]
        check(len(agents) == 1, 'ROCr UUID/BDF disagrees with KFD')
        r = agents[0]
        packages = [p for p in smi if p['bdf'] == k['physical_bdf']]
        check(len(packages) == 1, 'KFD/render-device package not identified by AMD SMI')
        check(h['cus'] == r['cus'] == k['simd_count'] // 4 == 228 // count, 'CU count mismatch')
        check(k['num_xcc'] == 6 // count, 'XCC count mismatch')
        check(k['numa_node'] >= 0 and 'MI300A' in h['name'], 'missing locality or wrong accelerator')
        result.append(dict(**h, uuid=r['uuid'], kfd_node=k['node'], unique_id=k['unique_id'], render_minor=k['render_minor'],
                           package_bdf=packages[0]['bdf'], package_uuid=packages[0]['uuid'], numa_node=k['numa_node']))
    for p in smi:
        check(sum(r['package_bdf'] == p['bdf'] for r in result) == count, 'incomplete physical package')
    return sorted(result, key=lambda r: (r['package_bdf'], r['bdf']))

def clean_env(xnack):
    """Copy the environment, clear GPU-visibility masks, and set HSA_XNACK."""
    env = os.environ.copy()
    for key in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
        env.pop(key, None)
    env['HSA_XNACK'] = str(xnack)
    return env


def capture(directory, binary, mode):
    """Snapshot HIP/ROCr/KFD/AMD SMI/NUMA topology and return the verified device list."""
    directory.mkdir(parents=True, exist_ok=True)

    def command(name, cmd, x=1):
        r = subprocess.run(cmd, env=clean_env(x), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        (directory / (name + '.txt')).write_text(r.stdout)
        (directory / (name + '.stderr')).write_text(r.stderr)
        check(r.returncode == 0, f'provenance command failed: {cmd}')
        return r.stdout

    partition = command('partition', ['amd-smi', 'static', '--partition'])
    check(re.findall(r'(?:COMPUTE|ACCELERATOR)_PARTITION:\s+(\w+)', partition) == [mode.upper()] * 2, 'compute mode mismatch')
    check(re.findall(r'MEMORY_PARTITION:\s+(\w+)', partition) == ['NPS1'] * 2, 'NPS1 not verified')
    smi = smi_devices(json.loads(command('smi_list', ['amd-smi', 'list', '--json'])))
    command('asic', ['amd-smi', 'static', '--asic'])
    command('smi_topology', ['amd-smi', 'topology'])
    command('smi_version', ['amd-smi', 'version'])
    command('hipconfig', ['hipconfig', '--full'])
    command('uname', ['uname', '-a'])
    command('numa', ['numactl', '--hardware'])
    command('lscpu', ['lscpu', '-J'])
    kfd = []
    for path in sorted(Path('/sys/class/kfd/kfd/topology/nodes').glob('*/properties')):
        raw = path.read_text()
        props = dict(line.split(maxsplit=1) for line in raw.splitlines() if len(line.split()) == 2)
        if int(props.get('simd_count', 0)) == 0:
            continue
        (directory / f'kfd_{path.parent.name}.txt').write_text(raw)
        minor = int(props['drm_render_minor'])
        render = Path(f'/sys/class/drm/renderD{minor}/device')
        (directory / f'kfd_{path.parent.name}_render.txt').write_text(str(render) + ' -> ' + str(render.resolve()) + '\n')
        package = render_package(props, mode, smi)
        kfd.append(dict(node=path.parent.name, domain=int(props.get('domain', 0)), location_id=int(props['location_id']), unique_id=int(props['unique_id']),
            simd_count=int(props['simd_count']), num_xcc=int(props['num_xcc']), render_minor=minor, **package))
    (directory / 'kfd_mapping.json').write_text(json.dumps(kfd, indent=2) + '\n')
    mappings = []
    for x in (0, 1):
        rocr = rocr_agents(command(f'rocminfo_xnack{x}', ['rocminfo'], x))
        check(all(r['xnack'] == ('xnack+' if x else 'xnack-') for r in rocr), 'XNACK not supported')
        hip = json.loads(command(f'hip_devices_xnack{x}', [str(binary), '--probe'], x))
        check(all(('xnack+' if x else 'xnack-') in h['arch'] for h in hip), 'HIP architecture XNACK mismatch')
        mappings.append(map_devices(mode, hip, rocr, kfd, smi))
    check([{k: v for k, v in r.items() if k != 'arch'} for r in mappings[0]] == [{k: v for k, v in r.items() if k != 'arch'} for r in mappings[1]], 'identity changes between XNACK settings')
    (directory / 'devices.json').write_text(json.dumps(mappings[1], indent=2) + '\n')
    return mappings[1]

def local_cores(numa_node, needed):
    """Return `needed` distinct local physical CPU cores from the allowed cpuset."""
    allowed = os.sched_getaffinity(0)
    selected = []
    seen = set()
    for cpu in sorted(allowed):
        p = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        if not (p / f'node{numa_node}').exists():
            continue
        key = ((p / 'topology/physical_package_id').read_text().strip(), (p / 'topology/core_id').read_text().strip())
        if key not in seen:
            selected.append(cpu)
            seen.add(key)
    check(len(selected) >= needed, 'not enough distinct local physical CPU cores in allowed cpuset')
    return selected[:needed]
