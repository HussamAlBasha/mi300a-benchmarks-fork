#!/usr/bin/env python3
"""Reduce shared intervals, retaining invalid trials. CPU-only, standard library."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st
from collections import defaultdict
from topology import check

def merge(intervals):
    """Merge a list of (start, end) intervals into sorted, non-overlapping spans."""
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def intersect(left, right):
    """Return the overlap intervals between two sorted, merged interval lists."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        a = max(left[i][0], right[j][0])
        b = min(left[i][1], right[j][1])
        if a < b:
            result.append((a, b))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def duration(intervals):
    """Total covered time of a set of intervals after merging overlaps."""
    return sum(b - a for a, b in merge(intervals))

def calibration_bounds(calibrations, markers):
    """Return the [lo, hi] host-to-trace clock offset bounds from bracketed markers."""
    check(len(calibrations) == 2, 'two bracketing clock markers required')
    lows = []
    highs = []
    for c in calibrations:
        times = [r['timestamp_ns'] for r in markers if r['label'] == c['label']]
        check(len(times) == 1, 'missing/duplicate clock marker')
        check(c['after_ns'] >= c['before_ns'], 'invalid marker bracket')
        lows.append(c['before_ns'] - times[0])
        highs.append(c['after_ns'] - times[0])
    lo, hi = max(lows), min(highs)
    check(lo <= hi, 'incompatible clock offsets / drift')
    check(hi - lo <= 100000, 'clock calibration uncertainty exceeds 100 us')
    return lo, hi

def trace_counts(kernels, bounds, t0, t1, expected):
    """Count complete/boundary TRIAD kernels in [t0, t1] and their guaranteed-busy intervals."""
    check(t1 > t0, 'nonpositive common interval')
    lo, hi = bounds
    check(len(kernels) == expected, 'missing/extra TRIAD dispatches')
    check(len({r['id'] for r in kernels}) == len(kernels), 'duplicate dispatch identity')
    complete = boundary = 0
    busy = []
    for k in kernels:
        a, b = k['start_ns'], k['end_ns']
        check(isinstance(a, int) and isinstance(b, int) and b > a, 'invalid kernel timestamps')
        if a + lo >= t0 and b + hi <= t1:
            complete += 1
        elif b + hi > t0 and a + lo < t1:
            boundary += 1
        # Guaranteed active time for any offset in the calibrated interval.
        left = max(t0, a + hi)
        right = min(t1, b + lo)
        if left < right:
            busy.append((left, right))
    check(complete > 0, 'no complete kernels in common interval')
    return complete, boundary, merge(busy)

def host_counts(batches, t0, t1):
    """Count complete/boundary launches from host-bracketed batches in [t0, t1]."""
    complete = boundary = 0
    previous = 0
    for b in batches:
        a, z, n = b['before_ns'], b['after_ns'], b['launches']
        check(z > a >= previous and n == 8, 'invalid/overlapping host batches')
        previous = z
        if a >= t0 and z <= t1:
            complete += n
        elif z > t0 and a < t1:
            boundary += n
    check(complete > 0, 'no complete host batches')
    return complete, boundary

def read_traces(directory):
    """Load TRIAD kernel dispatches and rq5_ ROCTx markers from a worker's trace CSVs."""
    kernels = []
    markers = []
    files = list(directory.rglob('*kernel_trace.csv'))
    check(len(files) == 1, 'expected one worker kernel trace')
    with files[0].open() as f:
        kernel_rows = list(csv.DictReader(f))
    for row in kernel_rows:
        if 'controlled_triad' in row['Kernel_Name']:
            kernels.append(dict(id=(row['Agent_Id'], row['Queue_Id'], row['Dispatch_Id']),
                start_ns=int(row['Start_Timestamp']), end_ns=int(row['End_Timestamp'])))
    check(len({k['id'][0] for k in kernels}) == 1, 'TRIAD dispatches on multiple agents')
    files = list(directory.rglob('*marker_api_trace.csv'))
    check(len(files) == 1, 'expected one worker marker trace')
    with files[0].open() as f:
        marker_rows = list(csv.DictReader(f))
    for row in marker_rows:
        if row['Function'].startswith('rq5_'):
            markers.append(dict(label=row['Function'], timestamp_ns=int(row['Start_Timestamp'])))
    return kernels, markers

def reduce_phase(root):
    """Validate one phase's workers and reduce traces/host batches into per-device rates."""
    spec = json.loads((root / 'phase.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    check(status['status'] == 'ok', 'phase failed: ' + status.get('reason', 'unknown'))
    check(set(status['workers']) == {d['worker'] for d in spec['devices']}, 'missing/unexpected process status')
    check(all(v == 0 for v in status['workers'].values()), 'worker exit failure')
    t0, t1 = spec['t0_ns'], spec['t1_ns']
    check(t1 - t0 == 2_000_000_000, 'measurement duration changed')
    all_busy = [(t0, t1)]
    rows = []
    check(len({d['bdf'] for d in spec['devices']}) == len(spec['devices']), 'duplicate device in phase')
    check(len({d['package_uuid'] for d in spec['devices']}) == 1, 'workers span packages')
    check(len({d['cpu'] for d in spec['devices']}) == len(spec['devices']), 'duplicate worker CPU')
    for dev in spec['devices']:
        worker = root / dev['worker']
        obj = json.loads((worker / 'result.json').read_text())
        check(obj['status'] == 'ok' and obj['validation_errors'] == 0, 'worker correctness failure')
        check(obj['worker'] == dev['worker'] and obj['rocr_mask'] == dev['uuid'], 'worker/mask mismatch')
        check(obj['device']['bdf'] == dev['bdf'] and obj['device']['cus'] == dev['cus'], 'worker identity mismatch')
        check(obj['affinity'] == [dev['cpu']], 'effective CPU affinity mismatch')
        check(obj['t0_ns'] == t0 and obj['t1_ns'] == t1, 'different measurement windows')
        for key, value in dict(elements=33554432, element_bytes=8, block_size=192, grid_blocks=174763, batch_size=8, warmup_launches=20, allocator=spec['allocator'], xnack=spec['xnack']).items():
            check(obj[key] == value, 'worker configuration mismatch: ' + key)
        check(sum(b['launches'] for b in obj['batches']) == obj['measured_phase_launches'], 'launch count mismatch')
        host_n, host_boundary = host_counts(obj['batches'], t0, t1)
        if spec['traced']:
            kernels, markers = read_traces(worker / 'trace')
            bounds = calibration_bounds(obj['calibrations'], markers)
            n, boundary, busy = trace_counts(kernels, bounds, t0, t1, obj['measured_phase_launches'] + 20)
            all_busy = intersect(all_busy, busy)
            duty = duration(busy) / (t1 - t0)
            uncertainty = bounds[1] - bounds[0]
        else:
            n, boundary, duty, uncertainty = host_n, host_boundary, None, None
        useful = 24 * obj['elements']
        seconds = (t1 - t0) / 1e9
        rows.append(dict(worker=dev['worker'], bdf=dev['bdf'], gb_s=n * useful / seconds / 1e9,
            useful_bytes=n * useful, boundary_bytes_upper=boundary * useful,
            boundary_gb_s_upper=boundary * useful / seconds / 1e9, host_gb_s=host_n * useful / seconds / 1e9,
            host_boundary_gb_s_upper=host_boundary * useful / seconds / 1e9,
            complete_kernels=n, active_fraction=duty, clock_uncertainty_ns=uncertainty))
    overlap = duration(all_busy) / (t1 - t0) if spec['traced'] else None
    check(not spec['traced'] or overlap >= .90, f'insufficient all-worker overlap: {overlap}')
    return dict(spec=spec, devices=rows, aggregate_gb_s=sum(r['useful_bytes'] for r in rows) / ((t1 - t0) / 1e9) / 1e9,
                boundary_gb_s_upper=sum(r['boundary_gb_s_upper'] for r in rows), all_worker_overlap=overlap)

def csv_write(path, rows):
    """Write a list of uniform dict rows to a CSV file (empty file when no rows)."""
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_phases(phases, expected_devices):
    """Pair each device's isolated baseline with its concurrent result and compute imbalance."""
    concurrent = [p for p in phases if p['spec']['condition'] == 'concurrent']
    isolated = [p for p in phases if p['spec']['condition'] == 'isolated']
    check(len(concurrent) == 1 and len(isolated) == len(expected_devices), 'missing/duplicate isolated or concurrent phase')
    c = concurrent[0]
    lookup = {}
    for p in isolated:
        check(len(p['devices']) == 1, 'isolated phase has multiple workers')
        row = p['devices'][0]
        check(row['bdf'] not in lookup, 'duplicate isolated baseline')
        lookup[row['bdf']] = row
        for key in ('allocator', 'xnack', 'traced', 'repeat', 'mode', 'node', 'package_uuid'):
            check(p['spec'][key] == c['spec'][key], 'unpaired ' + key)
    check(set(lookup) == set(expected_devices) == {r['bdf'] for r in c['devices']}, 'incomplete device pairing')
    rows = []
    for row in c['devices']:
        iso = lookup[row['bdf']]
        check(iso['gb_s'] > 0 and row['gb_s'] > 0, 'nonpositive paired rate')
        rows.append(dict(bdf=row['bdf'], isolated_gb_s=iso['gb_s'], concurrent_gb_s=row['gb_s'],
            slowdown=iso['gb_s'] / row['gb_s'], retained_fraction=row['gb_s'] / iso['gb_s']))
    rates = [r['concurrent_gb_s'] for r in rows]
    mean = st.mean(rates)
    return c, rows, dict(normalized_spread=(max(rates) - min(rates)) / mean, device_cv=st.pstdev(rates) / mean)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    root = args.campaign
    manifest = json.loads((root / 'campaign.json').read_text())
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    phases = []
    issues = []
    phase_rows = []
    for entry in manifest['schedule']:
        path = root / entry['path']
        try:
            result = reduce_phase(path)
            for key, value in entry.items():
                check(result['spec'][key] == value, 'phase differs from predeclared schedule: ' + key)
            for key in ('mode', 'node', 'package_uuid'):
                check(result['spec'][key] == manifest[key], 'phase differs from campaign: ' + key)
            phases.append(result)
            phase_rows.append(dict(path=entry['path'], status='accepted', reason='', aggregate_gb_s=result['aggregate_gb_s'], overlap=result['all_worker_overlap']))
        except (OSError, ValueError, KeyError, TypeError) as e:
            issues.append(f'{entry["path"]}: {e}')
            phase_rows.append(dict(path=entry['path'], status='invalid_or_missing', reason=str(e), aggregate_gb_s='', overlap=''))
    paired = []
    device_rows = []
    groups = defaultdict(list)
    for p in phases:
        s = p['spec']
        groups[(s['allocator'], s['xnack'], s['traced'], s['repeat'])].append(p)
    expected = {r['bdf'] for r in manifest['devices']}
    for key, group in sorted(groups.items()):
        try:
            c, rows, imbalance = pair_phases(group, expected)
            allocator, xnack, traced, repeat = key
            common = dict(mode=manifest['mode'], node=manifest['node'], package_uuid=manifest['package_uuid'], allocator=allocator, xnack=xnack, traced=traced, repeat=repeat)
            paired.append(dict(**common, aggregate_gb_s=c['aggregate_gb_s'], boundary_gb_s_upper=c['boundary_gb_s_upper'], overlap=c['all_worker_overlap'], **imbalance))
            for row in rows:
                device_rows.append(dict(**common, **row))
        except ValueError as e:
            issues.append(f'pair {key}: {e}')
    summary = []
    for alloc, xnack in manifest['configurations']:
        rows = [r for r in paired if (r['allocator'], r['xnack'], r['traced']) == (alloc, xnack, True)]
        complete = len(rows) == 6 and {r['repeat'] for r in rows} == set(range(1, 7))
        if not complete:
            issues.append(f'incomplete repeated cell {alloc}/{xnack}')
        values = [r['aggregate_gb_s'] for r in rows]
        summary.append(dict(allocator=alloc, xnack=xnack, valid_pairs=len(rows), complete=complete,
            mean_gb_s=st.mean(values) if complete else '', repeat_sd_gb_s=st.stdev(values) if complete else '',
            minimum_gb_s=min(values) if values else '', maximum_gb_s=max(values) if values else '',
            mean_spread=st.mean(r['normalized_spread'] for r in rows) if complete else '', mean_device_cv=st.mean(r['device_cv'] for r in rows) if complete else ''))
    device_summary = []
    for alloc, xnack in manifest['configurations']:
        for bdf in sorted(expected):
            rows = [r for r in device_rows if (r['allocator'], r['xnack'], r['traced'], r['bdf']) == (alloc, xnack, True, bdf)]
            complete = len(rows) == 6 and {r['repeat'] for r in rows} == set(range(1, 7))
            device_summary.append(dict(allocator=alloc, xnack=xnack, bdf=bdf, valid_pairs=len(rows), complete=complete,
                isolated_mean_gb_s=st.mean(r['isolated_gb_s'] for r in rows) if complete else '',
                concurrent_mean_gb_s=st.mean(r['concurrent_gb_s'] for r in rows) if complete else '',
                mean_paired_slowdown=st.mean(r['slowdown'] for r in rows) if complete else '',
                repeat_sd_slowdown=st.stdev(r['slowdown'] for r in rows) if complete else ''))
    overhead = []
    for untraced in (p for p in phases if not p['spec']['traced']):
        s = untraced['spec']
        ids = {d['bdf'] for d in untraced['devices']}
        traced = [p for p in phases if p['spec']['traced'] and p['spec']['allocator'] == s['allocator'] and p['spec']['xnack'] == s['xnack'] and p['spec']['condition'] == s['condition'] and {d['bdf'] for d in p['devices']} == ids]
        if traced:
            trace_host = st.mean(sum(d['host_gb_s'] for d in p['devices']) for p in traced)
            raw = sum(d['host_gb_s'] for d in untraced['devices'])
            overhead.append(dict(allocator=s['allocator'], xnack=s['xnack'], condition=s['condition'], devices=','.join(sorted(ids)),
                traced_host_mean_gb_s=trace_host, untraced_host_gb_s=raw, traced_over_untraced=trace_host / raw,
                note='same conservative host-batch estimator; one untraced sensitivity, order/time confounded'))
    csv_write(out / 'phases.csv', phase_rows)
    csv_write(out / 'paired_trials.csv', paired)
    csv_write(out / 'paired_devices.csv', device_rows)
    csv_write(out / 'summary.csv', summary)
    csv_write(out / 'tracing_sensitivity.csv', overhead)
    csv_write(out / 'device_summary.csv', device_summary)
    status_file = root / 'campaign_status.json'
    if not status_file.exists() or json.loads(status_file.read_text()).get('status') != 'complete':
        issues.append('campaign execution/preflight/postflight not complete')
    (out / 'validation.json').write_text(json.dumps(dict(accepted=not issues, issues=issues), indent=2) + '\n')
    print(f'{len(paired)} paired trials; {len(issues)} issues. ' + ('ACCEPTED' if not issues else 'INCOMPLETE / NOT ACCEPTED'))
    return 1 if issues else 0


if __name__ == '__main__':
    raise SystemExit(main())
