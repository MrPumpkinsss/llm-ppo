"""
baseline.py
"""

import numpy as np
from itertools import product
from simulator import simulate_inference_tpot, check_memory_feasibility
import time


def brute_force_optimal(layers, cluster):
    nl, nd = len(layers), len(cluster.devices)
    ss = nd ** nl
    if ss > 100_000_000:
        return None
    best_t, best_p, feas = float('inf'), None, 0
    t0 = time.time()
    for pt in product(range(nd), repeat=nl):
        p = list(pt)
        if not check_memory_feasibility(layers, cluster, p):
            continue
        feas += 1
        r = simulate_inference_tpot(layers, cluster, p)
        if r['tpot'] < best_t:
            best_t, best_p = r['tpot'], p[:]
    if not best_p:
        return None
    return {'partition': best_p, 'tpot': best_t,
            'result': simulate_inference_tpot(layers, cluster, best_p)}


def dp_optimal(layers, cluster, beam=500):
    nl, nd = len(layers), len(cluster.devices)
    beam_list = []
    for d in range(nd):
        if layers[0].param_size <= cluster.devices[d].memory:
            t = (layers[0].flops / cluster.devices[d].compute_power) * 1000
            m = [0.0] * nd; m[d] = layers[0].param_size
            beam_list.append((t, d, [d], tuple(m)))
    for li in range(1, nl):
        nxt = []
        for cost, ld, part, mt in beam_list:
            m = list(mt)
            for d in range(nd):
                if m[d] + layers[li].param_size > cluster.devices[d].memory:
                    continue
                ct = (layers[li].flops / cluster.devices[d].compute_power) * 1000
                comm = 0 if ld == d else (layers[li - 1].activation_size / cluster.bandwidth_matrix[ld][d]) * 1000 + 0.3
                nm = m[:]; nm[d] += layers[li].param_size
                nxt.append((cost + ct + comm, d, part + [d], tuple(nm)))
        nxt.sort(key=lambda x: x[0])
        per_d = {}
        for s in nxt:
            dd = s[1]
            if dd not in per_d: per_d[dd] = []
            if len(per_d[dd]) < beam // nd + 1: per_d[dd].append(s)
        beam_list = []
        for v in per_d.values(): beam_list.extend(v)
        beam_list.sort(key=lambda x: x[0])
        beam_list = beam_list[:beam]
    if not beam_list:
        return None
    best = min(beam_list, key=lambda x: x[0])
    r = simulate_inference_tpot(layers, cluster, best[2])
    return {'partition': best[2], 'tpot': r['tpot'], 'result': r}


def greedy_baseline(layers, cluster):
    nl, nd = len(layers), len(cluster.devices)
    tc = sum(d.compute_power for d in cluster.devices)
    part, idx = [], 0
    for di in range(nd):
        n = nl - idx if di == nd - 1 else max(1, min(nl - idx, round(nl * cluster.devices[di].compute_power / tc)))
        part += [di] * n; idx += n
    part = (part + [nd - 1] * nl)[:nl]
    r = simulate_inference_tpot(layers, cluster, part)
    return {'partition': part, 'tpot': r['tpot'], 'result': r}


def greedy_memory_aware(layers, cluster):
    nl, nd = len(layers), len(cluster.devices)
    part, dm = [], [0.0] * nd
    for li in range(nl):
        best_d, best_s = None, float('inf')
        for d in range(nd):
            if dm[d] + layers[li].param_size > cluster.devices[d].memory:
                continue
            ct = (layers[li].flops / cluster.devices[d].compute_power) * 1000
            comm = 0
            if part and part[-1] != d:
                comm = (layers[li - 1].activation_size / cluster.bandwidth_matrix[part[-1]][d]) * 1000 + 0.3
            if ct + comm < best_s:
                best_s, best_d = ct + comm, d
        if best_d is None:
            best_d = max(range(nd), key=lambda d: cluster.devices[d].memory - dm[d])
        part.append(best_d); dm[best_d] += layers[li].param_size
    r = simulate_inference_tpot(layers, cluster, part)
    return {'partition': part, 'tpot': r['tpot'], 'result': r}


def uniform_baseline(layers, cluster):
    nl, nd = len(layers), len(cluster.devices)
    per, rem = nl // nd, nl % nd
    part = []
    for d in range(nd):
        part += [d] * (per + (1 if d < rem else 0))
    r = simulate_inference_tpot(layers, cluster, part)
    return {'partition': part, 'tpot': r['tpot'], 'result': r}


def random_search_baseline(layers, cluster, n_trials=50000, seed=42):
    rng = np.random.RandomState(seed)
    nl, nd = len(layers), len(cluster.devices)
    best_t, best_p = float('inf'), None
    for trial in range(n_trials):
        if trial < n_trials * 0.5:
            p = rng.randint(0, nd, nl).tolist()
        elif trial < n_trials * 0.8:
            n_segs = rng.randint(nd, nd * 2 + 1)
            cuts = sorted(rng.choice(range(1, nl), min(n_segs - 1, nl - 1), False))
            cuts = [0] + list(cuts) + [nl]
            p = []
            for i in range(len(cuts) - 1):
                p += [rng.randint(0, nd)] * (cuts[i + 1] - cuts[i])
        else:
            if best_p:
                p = best_p[:]
                for _ in range(rng.randint(1, max(2, nl // 4))):
                    p[rng.randint(0, nl)] = rng.randint(0, nd)
            else:
                p = rng.randint(0, nd, nl).tolist()
        if not check_memory_feasibility(layers, cluster, p):
            continue
        r = simulate_inference_tpot(layers, cluster, p)
        if r['tpot'] < best_t:
            best_t, best_p = r['tpot'], p[:]
    if not best_p:
        return None
    return {'partition': best_p, 'tpot': best_t,
            'result': simulate_inference_tpot(layers, cluster, best_p)}