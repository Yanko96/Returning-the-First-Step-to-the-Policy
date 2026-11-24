#!/usr/bin/env python
"""OP reference ILS solver, with an alpha-collapse self-check.

Settings: see the paper's appendix.
Pure numpy, CPU only. Kool-convention OP instances (depot+nodes ~ U([0,1]^2),
distance prizes p_i=(1+floor(99*d0i/max d0j))/100), budgets = alpha * L_TSP(instance)
for alpha in {0.30..0.80}, plus 'std' (Kool fixed T: n50=3.0, n100=4.0) and
'cal' = 1.2*L_TSP (calibration arm, visit-all certificate exists by construction).

Solver: randomized greedy insertion -> [2-opt/Or-opt compress -> greedy ADD ->
in-place SWAP(1-1)] descent to fixed point -> ILS (drop 15%, re-descend) x6,
16 restarts per (instance, budget).

Usage:
  python op_ils_reference.py            # full run
  python op_ils_reference.py --smoke    # quick code check (99000-series)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

EPS = 1e-12
ALPHAS = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
SCALES = {50: dict(T_std=3.0, inst_seed0=30000), 100: dict(T_std=4.0, inst_seed0=31000)}
N_INST = 24
N_RESTARTS = 16
ILS_ROUNDS = 160         # no early stop: deterministic effort per restart
ILS_PATIENCE = 10 ** 9
WALK_TOL = 0.01          # accept kicks up to 1% (of elite prize) worse: random-walk bridging
DROP_FRAC_SMALL = (0.10, 0.30)
DROP_FRAC_BIG = (0.40, 0.60)   # big kick (near-fresh sample w/ memory)
FRESH_EVERY = 6                # every 6th round: full fresh construction (basin mixing)
CONSTRUCT_NOISE = 0.15
REFILL_NOISE = 0.20
# construction-criterion shapes (gp, gd): score = prize^gp / detour^gd — greedy-bias diversity
GAMMA_SET = [(1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (4.0, 0.5), (1.0, 0.2)]


# ---------------- instance generation (Kool conventions) ----------------

def gen_instance(n, seed):
    rng = np.random.default_rng(seed)
    depot = rng.random(2)
    loc = rng.random((n, 2))
    d0 = np.linalg.norm(loc - depot, axis=1)
    prize = (1.0 + np.floor(99.0 * d0 / d0.max())) / 100.0
    return depot, loc, prize


def dist_matrix(depot, loc):
    pts = np.vstack([depot[None, :], loc])  # index 0 = depot
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def tour_len(D, t):
    return float(D[t, np.roll(t, -1)].sum())


# ---------------- tour improvement (cycle, depot at t[0]) ----------------

def two_opt(D, t, max_iter=400):
    t = t.copy()
    for _ in range(max_iter):
        m = len(t)
        if m < 4:
            return t
        succ = np.roll(t, -1)
        base = D[t, succ]
        delta = D[np.ix_(t, t)] + D[np.ix_(succ, succ)] - base[:, None] - base[None, :]
        iu = np.triu_indices(m, k=2)
        vals = delta[iu]
        k = int(np.argmin(vals))
        if vals[k] < -1e-10:
            i, j = int(iu[0][k]), int(iu[1][k])
            t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
        else:
            return t
    return t


def or_opt(D, t, max_iter=250):
    t = t.copy()
    for _ in range(max_iter):
        m = len(t)
        if m < 5:
            return t
        succ = np.roll(t, -1)
        base = D[t, succ]
        P = D[np.ix_(t, t)]
        best_delta = -1e-10
        best_move = None
        for L in (1, 2, 3):
            smax = m - L
            if smax < 1:
                continue
            s_arr = np.arange(1, smax + 1)
            seg_last = t[s_arr + L - 1]
            pred = t[s_arr - 1]
            succ_seg = t[(s_arr + L) % m]
            gain_rem = D[pred, t[s_arr]] + D[seg_last, succ_seg] - D[pred, succ_seg]
            cost_ins = P[:, s_arr].T + D[np.ix_(seg_last, succ)] - base[None, :]
            delta = cost_ins - gain_rem[:, None]
            jgrid = np.arange(m)[None, :]
            sgrid = s_arr[:, None]
            invalid = (jgrid >= sgrid - 1) & (jgrid <= sgrid + L - 1)
            delta = np.where(invalid, np.inf, delta)
            si, j = np.unravel_index(int(np.argmin(delta)), delta.shape)
            if delta[si, j] < best_delta:
                best_delta = float(delta[si, j])
                best_move = (int(s_arr[si]), L, int(j))
        if best_move is None:
            return t
        s, L, j = best_move
        seg = t[s:s + L]
        rest = np.concatenate([t[:s], t[s + L:]])
        jj = j if j < s else j - L
        t = np.concatenate([rest[:jj + 1], seg, rest[jj + 1:]])
    return t


def compress(D, t):
    """Alternate 2-opt / Or-opt until neither improves."""
    L = tour_len(D, t)
    for _ in range(6):
        t = two_opt(D, t)
        t = or_opt(D, t)
        L2 = tour_len(D, t)
        if L2 > L - 1e-12:
            return t, L2
        L = L2
    return t, L


# ---------------- TSP normalizer ----------------

def nn_tour(D, rng=None):
    m = D.shape[0]
    unv = np.ones(m, bool)
    unv[0] = False
    t = [0]
    cur = 0
    for _ in range(m - 1):
        cand = np.where(unv)[0]
        dd = D[cur, cand]
        if rng is not None:
            dd = dd * rng.uniform(1.0, 1.3, size=len(cand))
        nxt = int(cand[np.argmin(dd)])
        t.append(nxt)
        unv[nxt] = False
        cur = nxt
    return np.array(t, dtype=int)


def tsp_normalizer(D, seed, n_starts=4):
    rng = np.random.default_rng(seed)
    best = np.inf
    for s in range(n_starts):
        t = nn_tour(D, None if s == 0 else rng)
        t, L = compress(D, t)
        if L < best:
            best = L
    return float(best)


# ---------------- OP solver ----------------

def op_insert_loop(D, prize, t, L, unv, B, rng=None, noise=0.0, gp=1.0, gd=1.0):
    """Greedy (optionally noised) insertion, score = prize^gp / detour^gd, until no feasible add."""
    added = False
    while True:
        cand = np.where(unv)[0]
        if cand.size == 0:
            break
        succ = np.roll(t, -1)
        det = D[np.ix_(t, cand)] + D[np.ix_(cand, succ)].T - D[t, succ][:, None]
        det = np.maximum(det, 0.0)
        feas = (L + det) <= B + EPS
        if not feas.any():
            break
        score = (prize[cand][None, :] ** gp) / ((det + 1e-9) ** gd)
        if rng is not None and noise > 0:
            score = score * rng.uniform(1.0 - noise, 1.0 + noise, size=score.shape)
        score = np.where(feas, score, -np.inf)
        k, c = np.unravel_index(int(np.argmax(score)), score.shape)
        j = int(cand[c])
        L += float(det[k, c])
        t = np.insert(t, k + 1, j)
        unv[j] = False
        added = True
    return t, L, unv, added


def op_swap_once(D, prize, t, L, unv, B):
    """Best in-place 1-1 swap with positive prize gain; returns applied flag."""
    if len(t) < 2:
        return t, L, unv, False
    cand = np.where(unv)[0]
    if cand.size == 0:
        return t, L, unv, False
    pred = np.roll(t, 1)
    succ = np.roll(t, -1)
    dl = D[np.ix_(pred, cand)] + D[np.ix_(cand, succ)].T \
        - (D[pred, t] + D[t, succ])[:, None]
    pg = prize[cand][None, :] - prize[t][:, None]
    # accept prize-improving swaps, or equal-prize swaps that strictly shorten
    valid = ((L + dl) <= B + EPS) & ((pg > 1e-12) | ((np.abs(pg) <= 1e-12) & (dl < -1e-10)))
    valid[0, :] = False  # never swap out the depot
    if not valid.any():
        return t, L, unv, False
    score = np.where(valid, pg - 1e-6 * dl, -np.inf)
    k, c = np.unravel_index(int(np.argmax(score)), score.shape)
    j = int(cand[c])
    unv[t[k]] = True
    unv[j] = False
    L += float(dl[k, c])
    t = t.copy()
    t[k] = j
    return t, L, unv, True


def op_descend(D, prize, t, unv, B, max_rounds=40):
    L = tour_len(D, t)
    for _ in range(max_rounds):
        changed = False
        t2, L2 = compress(D, t)
        if L2 < L - 1e-12:
            changed = True
        t, L = t2, L2
        t, L, unv, added = op_insert_loop(D, prize, t, L, unv, B)
        changed = changed or added
        for _ in range(60):
            t, L, unv, sw = op_swap_once(D, prize, t, L, unv, B)
            if not sw:
                break
            changed = True
        if not changed:
            break
    return t, L, unv


def _fresh_start(D, prize, B, rng, shape=None):
    m = D.shape[0]
    t = np.array([0], dtype=int)
    unv = np.ones(m, bool)
    unv[0] = False
    if shape is None:
        gp, gd = GAMMA_SET[rng.integers(len(GAMMA_SET))]
        noise = CONSTRUCT_NOISE
    else:
        gp, gd = shape  # deterministic portfolio member
        noise = 0.0
    t, L, unv, _ = op_insert_loop(D, prize, t, 0.0, unv, B, rng=rng,
                                  noise=noise, gp=gp, gd=gd)
    return op_descend(D, prize, t, unv, B)


def op_solve_restart(D, prize, B, seed):
    rng = np.random.default_rng(seed)
    m = D.shape[0]
    # deterministic gamma-portfolio starts: one per criterion shape, elite = best
    best_p, best_t, best_L = -1.0, None, None
    for shape in GAMMA_SET:
        t2, L2, unv2 = _fresh_start(D, prize, B, rng, shape=shape)
        p2 = float(prize[t2].sum())
        if p2 > best_p + 1e-12 or (abs(p2 - best_p) <= 1e-12 and best_L is not None
                                   and L2 < best_L - 1e-12):
            best_p, best_t, best_L = p2, t2.copy(), L2
    t = best_t.copy()
    unv = np.ones(m, bool)
    unv[t] = False
    for rd in range(ILS_ROUNDS):
        vis = t[1:]
        if len(vis) == 0:
            break
        if rd % FRESH_EVERY == FRESH_EVERY - 1:
            # full fresh construction: independent basin sample
            t2, L2, unv2 = _fresh_start(D, prize, B, rng)
        else:
            mode = rd % 3
            if mode == 2:      # big kick: near-fresh sample around current walk point
                frac = rng.uniform(*DROP_FRAC_BIG)
                ndrop = min(max(1, int(round(frac * len(vis)))), len(vis))
                drop = rng.choice(vis, size=ndrop, replace=False)
            elif mode == 1:    # targeted kick: evict worst prize/saving nodes (noised)
                pred = np.roll(t, 1)
                succ = np.roll(t, -1)
                sav = D[pred, t] + D[t, succ] - D[pred, succ]
                ratio = prize[t] / (sav + 1e-9)
                ratio[0] = np.inf  # depot never dropped
                noised = ratio * rng.uniform(0.6, 1.4, size=len(ratio))
                q = min(rng.integers(3, 6), len(vis))
                drop = t[np.argsort(noised)[:q]]
            else:              # small spatial kick: contiguous tour segment or subset
                frac = rng.uniform(*DROP_FRAC_SMALL)
                ndrop = min(max(1, int(round(frac * len(vis)))), len(vis))
                if rd % 2 == 0:
                    drop = rng.choice(vis, size=ndrop, replace=False)
                else:
                    u = rng.integers(0, len(vis))
                    drop = vis[(u + np.arange(ndrop)) % len(vis)]
            t2 = t[~np.isin(t, drop)]
            unv2 = np.ones(m, bool)
            unv2[t2] = False
            # noised refill (diversifies the deterministic descent), then descend
            t2, L2, unv2, _ = op_insert_loop(D, prize, t2, tour_len(D, t2), unv2, B,
                                             rng=rng, noise=REFILL_NOISE)
            t2, L2, unv2 = op_descend(D, prize, t2, unv2, B)
        p2 = float(prize[t2].sum())
        if p2 > best_p + 1e-12 or (abs(p2 - best_p) <= 1e-12 and L2 < best_L - 1e-12):
            best_p, best_t, best_L = p2, t2.copy(), L2
        # walk acceptance: stay on t2 if within 2% of elite; else reset to elite
        if p2 >= best_p * (1.0 - 2.0 * WALK_TOL) - 1e-12:
            t, unv = t2, unv2
        else:
            t = best_t.copy()
            unv = np.ones(m, bool)
            unv[t] = False
    return best_p, best_t, best_L


def solve_cell(D, prize, B, seeds):
    """Multi-restart solve; returns best + per-restart records."""
    best = (-1.0, None, None)
    restart_prizes = []
    restart_sets = []
    for sd in seeds:
        p, t, L = op_solve_restart(D, prize, B, sd)
        assert L <= B + 1e-9, f"INFEASIBLE tour: L={L} > B={B}"
        assert t[0] == 0 and len(np.unique(t)) == len(t), "malformed tour"
        restart_prizes.append(p)
        restart_sets.append(sorted(int(x) for x in t[1:]))
        if p > best[0] + 1e-12 or (abs(p - best[0]) <= 1e-12 and best[2] is not None and L < best[2] - 1e-12):
            best = (p, t.copy(), L)
    return best, restart_prizes, restart_sets


# ---------------- driver ----------------

def run(smoke=False):
    t0 = time.time()
    n_inst = 2 if smoke else N_INST
    restarts = 4 if smoke else N_RESTARTS
    alphas = [0.40, 0.60] if smoke else ALPHAS
    results = {}
    for n, cfg in SCALES.items():
        scale_res = []
        for i in range(n_inst):
            ti = time.time()
            inst_seed = (99000 + i if n == 50 else 99100 + i) if smoke \
                else cfg["inst_seed0"] + i
            gidx = (0 if n == 50 else N_INST) + i
            depot, loc, prize = gen_instance(n, inst_seed)
            D = dist_matrix(depot, loc)
            prize_full = np.concatenate([[0.0], prize])  # align with D (index 0 = depot)
            Ltsp = tsp_normalizer(D, 50000 + gidx)
            budgets = {f"a{a:.2f}": a * Ltsp for a in alphas}
            budgets["std"] = cfg["T_std"]
            budgets["cal"] = 1.2 * Ltsp
            seeds = [40000 + gidx * 100 + r for r in range(restarts)]
            rec = dict(instance_seed=inst_seed, L_tsp=Ltsp,
                       alpha_std=cfg["T_std"] / Ltsp,
                       total_prize=float(prize.sum()), cells={})
            for label, B in budgets.items():
                (bp, bt, bL), rp, rs = solve_cell(D, prize_full, B, seeds)
                cell = dict(budget=float(B),
                            best_prize=bp,
                            prize_ratio=bp / rec["total_prize"],
                            visit_rate=(len(bt) - 1) / n,
                            tour_len=bL,
                            restart_prizes=rp,
                            best_visited=sorted(int(x) for x in bt[1:]))
                if label in ("std", "cal"):
                    cell["restart_sets"] = rs
                rec["cells"][label] = cell
            scale_res.append(rec)
            print(f"[n={n} inst {i:2d}] L_tsp={Ltsp:.3f} a_std={rec['alpha_std']:.3f} "
                  f"std v={rec['cells']['std']['visit_rate']:.2f} "
                  f"p={rec['cells']['std']['prize_ratio']:.2f} "
                  f"cal v={rec['cells']['cal']['visit_rate']:.2f} "
                  f"({time.time() - ti:.1f}s)", flush=True)
            if i == 0:
                proj = (time.time() - ti) * n_inst
                print(f"  projected for n={n}: ~{proj / 60:.1f} min", flush=True)
        results[n] = scale_res
    print(f"\nTOTAL solve time: {(time.time() - t0) / 60:.1f} min", flush=True)
    return results


# ---------------- metrics + checks ----------------

def interp_alpha_sat(alphas, vmeans, thresh=0.95):
    for k, (a, v) in enumerate(zip(alphas, vmeans)):
        if v >= thresh:
            if k == 0:
                return a
            a0, v0 = alphas[k - 1], vmeans[k - 1]
            return a0 + (a - a0) * (thresh - v0) / max(v - v0, 1e-12)
    return None  # > max alpha


def run_checks(results, smoke=False):
    out = dict(scales={}, tool_checks={}, criteria={}, status=None)
    mono_viol, mono_tot = 0, 0
    all_pass = True
    tool_ok = True
    for n, recs in results.items():
        S = {}
        S["alpha_std_mean"] = float(np.mean([r["alpha_std"] for r in recs]))
        vmeans = [float(np.mean([r["cells"][f"a{a:.2f}"]["visit_rate"] for r in recs])) for a in ALPHAS] \
            if not smoke else []
        pmeans = [float(np.mean([r["cells"][f"a{a:.2f}"]["prize_ratio"] for r in recs])) for a in ALPHAS] \
            if not smoke else []
        S["v_grid"] = dict(zip([f"{a:.2f}" for a in ALPHAS], vmeans))
        S["p_grid"] = dict(zip([f"{a:.2f}" for a in ALPHAS], pmeans))
        S["v_std"] = float(np.mean([r["cells"]["std"]["visit_rate"] for r in recs]))
        S["p_std"] = float(np.mean([r["cells"]["std"]["prize_ratio"] for r in recs]))
        # calibration-arm certificate (visit-all, full prize)
        cal_ok_count = sum(1 for r in recs
                           if r["cells"]["cal"]["visit_rate"] >= 0.99 - 1e-9
                           and r["cells"]["cal"]["prize_ratio"] >= 0.99 - 1e-9)
        S["cal_ok_count"] = cal_ok_count
        # monotonicity (accumulated globally)
        if not smoke:
            for r in recs:
                ps = [r["cells"][f"a{a:.2f}"]["prize_ratio"] for a in ALPHAS]
                for a_prev, a_next in zip(ps[:-1], ps[1:]):
                    mono_tot += 1
                    if a_next < a_prev - 0.005:
                        mono_viol += 1
        # noise floor at std: half-split cell-best noise;
        # strict per-restart range kept for reference
        spreads, half_diffs = [], []
        for r in recs:
            rp = np.array(r["cells"]["std"]["restart_prizes"])
            spreads.append((rp.max() - rp.min()) / max(rp.max(), 1e-9))
            h = len(rp) // 2
            half_diffs.append(abs(rp[:h].max() - rp[h:].max()) / max(rp.max(), 1e-9))
        S["strict_spread"] = float(np.mean(spreads))
        S["half_split_noise"] = float(np.mean(half_diffs))
        S["half_split_worst"] = float(np.max(half_diffs))
        # near-optima subset diversity at std
        div, ndist = [], []
        for r in recs:
            rp = np.array(r["cells"]["std"]["restart_prizes"])
            sets = [frozenset(s) for s, p in zip(r["cells"]["std"]["restart_sets"], rp)
                    if p >= 0.995 * rp.max()]
            uniq = list(set(sets))
            ndist.append(len(uniq))
            if len(uniq) > 1:
                jd = []
                for x in range(len(uniq)):
                    for y in range(x + 1, len(uniq)):
                        a_, b_ = uniq[x], uniq[y]
                        jd.append(1.0 - len(a_ & b_) / max(len(a_ | b_), 1))
                div.append(float(np.mean(jd)))
            else:
                div.append(0.0)
        S["diversity_jaccard_mean"] = float(np.mean(div))
        S["diversity_distinct_mean"] = float(np.mean(ndist))
        if not smoke:
            a_sat = interp_alpha_sat(ALPHAS, vmeans)
            S["alpha_sat"] = a_sat if a_sat is not None else ">0.80"
            # calibration criteria
            C = {}
            C["std_visit_rate_in_band"] = 0.40 <= S["v_std"] <= 0.75
            if a_sat is not None:
                margin = a_sat - S["alpha_std_mean"]
                S["alpha_sat_margin"] = margin
            else:
                margin = 0.80 - S["alpha_std_mean"]  # strict lower bound
                S["alpha_sat_margin"] = f">{margin:.2f}"
            C["alpha_sat_margin_ok"] = margin >= 0.15
            C["prize_spread_ok"] = (S["p_grid"]["0.60"] - S["p_grid"]["0.40"]) >= 0.06
            S["prize_spread_pp"] = 100 * (S["p_grid"]["0.60"] - S["p_grid"]["0.40"])
            C["visit_spread_ok"] = (S["v_grid"]["0.60"] - S["v_grid"]["0.40"]) >= 0.10
            S["visit_spread_pp"] = 100 * (S["v_grid"]["0.60"] - S["v_grid"]["0.40"])
            pinned = sum(1 for r in recs
                         if not (0.25 <= r["cells"]["std"]["visit_rate"] <= 0.90))
            C["few_pinned_instances"] = pinned <= 4
            S["pinned_count"] = pinned
            cal_ok = cal_ok_count >= 22
            half_ok = S["half_split_noise"] <= 0.01
            S["cal_certificate_ok"] = cal_ok
            S["half_split_noise_ok"] = half_ok
            tool_ok = tool_ok and cal_ok and half_ok
            all_pass = all_pass and all(C.values())
            out["criteria"][n] = C
        out["scales"][n] = S
    if not smoke:
        mono_ok = (mono_viol / max(mono_tot, 1)) <= 0.05
        out["tool_checks"] = dict(monotone_violations=mono_viol, monotone_total=mono_tot,
                                  monotone_ok=mono_ok,
                                  cal_certificate_per_scale={n: out["scales"][n]["cal_certificate_ok"] for n in results},
                                  half_split_noise_per_scale={n: out["scales"][n]["half_split_noise_ok"] for n in results})
        tool_ok = tool_ok and mono_ok
        out["tool_valid"] = tool_ok
        out["status"] = ("NOT EVALUATED (tool checks failed)" if not tool_ok
                         else ("PASS" if all_pass else "FAIL"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    mode = "SMOKE (quick code check)" if args.smoke else "FULL RUN"
    print(f"=== OP reference ILS self-check — {mode} ===", flush=True)
    results = run(smoke=args.smoke)
    status = run_checks(results, smoke=args.smoke)
    print("\n" + "=" * 70)
    print(json.dumps(status, indent=2, ensure_ascii=False, default=str))
    if not args.smoke:
        out_root = os.environ.get("OUTPUT_ROOT", "outputs")
        os.makedirs(out_root, exist_ok=True)
        dst = os.path.join(out_root, "op_ils_reference_result.json")
        with open(dst, "w") as f:
            json.dump(dict(results={str(k): v for k, v in results.items()},
                           status=status), f, default=str)
        print("\nwritten:", dst, flush=True)
        print(f"\n=== SELF-CHECK STATUS: {status['status']} ===", flush=True)


if __name__ == "__main__":
    main()
