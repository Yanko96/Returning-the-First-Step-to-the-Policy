"""The reference ILS solver with an optional forced-first-node lock (t[1] == s).

Used by the start-stakes instrument, which compares the best tour reachable from a
free start against the best reachable from a forced one.

Every operator that can move index 1 is constrained. They are enumerated explicitly
because this class of lock fails SILENTLY when one is missed:

  two_opt        reverses t[i+1:j+1]; i=0 would move t[1]      -> require i >= 1
  or_opt         segment start s=1 contains t[1]               -> require s >= 2
                 insertion after rest[jj] with jj=0 lands at 1 -> require jj >= 1
  op_insert_loop np.insert(t, k+1, j) with k=0 lands at 1      -> mask row k=0
  op_swap_once   t[k]=j replaces t[1] when k=1                 -> valid[1,:] = False
                 (the original already sets valid[0,:] = False for the depot)
  ILS kicks      vis = t[1:] can drop s                        -> vis = t[2:]
                 targeted eviction sets ratio[0]=inf only      -> also ratio[1]=inf
  _fresh_start   starts from [0]                               -> start from [0, s]
  compress       delegates to two_opt / or_opt                 -> covered above

With forced=None every function must reduce to the original op_ils_reference bit-for-bit;
selftest_free_equivalence asserts that against the unlocked reference.
"""
import numpy as np

import op_ils_reference as E

EPS = E.EPS
ILS_ROUNDS = E.ILS_ROUNDS
WALK_TOL = E.WALK_TOL
DROP_FRAC_SMALL = E.DROP_FRAC_SMALL
DROP_FRAC_BIG = E.DROP_FRAC_BIG
FRESH_EVERY = E.FRESH_EVERY
CONSTRUCT_NOISE = E.CONSTRUCT_NOISE
REFILL_NOISE = E.REFILL_NOISE
GAMMA_SET = E.GAMMA_SET
tour_len = E.tour_len


def _chk(t, forced):
    if forced is not None:
        assert len(t) >= 2 and int(t[1]) == int(forced), "LOCK BROKEN: t[1] != forced"


def two_opt(D, t, max_iter=400, forced=None):
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
        if forced is not None:
            vals = np.where(iu[0] >= 1, vals, np.inf)   # i=0 would reverse t[1:]
        k = int(np.argmin(vals))
        if vals[k] < -1e-10:
            i, j = int(iu[0][k]), int(iu[1][k])
            t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
        else:
            return t
    return t


def or_opt(D, t, max_iter=250, forced=None):
    t = t.copy()
    lo = 2 if forced is not None else 1
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
            if smax < lo:
                continue
            s_arr = np.arange(lo, smax + 1)
            if s_arr.size == 0:
                continue
            seg_last = t[s_arr + L - 1]
            pred = t[s_arr - 1]
            succ_seg = t[(s_arr + L) % m]
            gain_rem = D[pred, t[s_arr]] + D[seg_last, succ_seg] - D[pred, succ_seg]
            cost_ins = P[:, s_arr].T + D[np.ix_(seg_last, succ)] - base[None, :]
            delta = cost_ins - gain_rem[:, None]
            jgrid = np.arange(m)[None, :]
            sgrid = s_arr[:, None]
            invalid = (jgrid >= sgrid - 1) & (jgrid <= sgrid + L - 1)
            if forced is not None:
                invalid = invalid | (jgrid < 1)   # jj=0 would land the segment at 1
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


def compress(D, t, forced=None):
    L = tour_len(D, t)
    for _ in range(6):
        t = two_opt(D, t, forced=forced)
        t = or_opt(D, t, forced=forced)
        L2 = tour_len(D, t)
        if L2 > L - 1e-12:
            return t, L2
        L = L2
    return t, L


def op_insert_loop(D, prize, t, L, unv, B, rng=None, noise=0.0, gp=1.0, gd=1.0,
                   forced=None):
    added = False
    while True:
        cand = np.where(unv)[0]
        if cand.size == 0:
            break
        succ = np.roll(t, -1)
        det = D[np.ix_(t, cand)] + D[np.ix_(cand, succ)].T - D[t, succ][:, None]
        det = np.maximum(det, 0.0)
        feas = (L + det) <= B + EPS
        if forced is not None and feas.shape[0] > 0:
            feas = feas.copy()
            feas[0, :] = False                   # k=0 would insert at index 1
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


def op_swap_once(D, prize, t, L, unv, B, forced=None):
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
    valid = ((L + dl) <= B + EPS) & ((pg > 1e-12) | ((np.abs(pg) <= 1e-12) & (dl < -1e-10)))
    valid[0, :] = False                          # never swap out the depot
    if forced is not None and valid.shape[0] > 1:
        valid[1, :] = False                      # never swap out the forced start
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


def op_descend(D, prize, t, unv, B, max_rounds=40, forced=None):
    L = tour_len(D, t)
    for _ in range(max_rounds):
        changed = False
        t2, L2 = compress(D, t, forced=forced)
        if L2 < L - 1e-12:
            changed = True
        t, L = t2, L2
        _chk(t, forced)
        t, L, unv, added = op_insert_loop(D, prize, t, L, unv, B, forced=forced)
        changed = changed or added
        _chk(t, forced)
        for _ in range(60):
            t, L, unv, sw = op_swap_once(D, prize, t, L, unv, B, forced=forced)
            if not sw:
                break
            changed = True
        _chk(t, forced)
        if not changed:
            break
    return t, L, unv


def _fresh_start(D, prize, B, rng, shape=None, forced=None):
    m = D.shape[0]
    unv = np.ones(m, bool)
    unv[0] = False
    if forced is None:
        t = np.array([0], dtype=int)
        L0 = 0.0
    else:
        t = np.array([0, int(forced)], dtype=int)
        unv[int(forced)] = False
        L0 = tour_len(D, t)
        if L0 > B + EPS:
            return None, None, None
    if shape is None:
        gp, gd = GAMMA_SET[rng.integers(len(GAMMA_SET))]
        noise = CONSTRUCT_NOISE
    else:
        gp, gd = shape
        noise = 0.0
    t, L, unv, _ = op_insert_loop(D, prize, t, L0, unv, B, rng=rng,
                                  noise=noise, gp=gp, gd=gd, forced=forced)
    return op_descend(D, prize, t, unv, B, forced=forced)


def op_solve_restart(D, prize, B, seed, forced=None):
    """Returns (best_prize, best_tour, best_len). forced=None reproduces op_ils_reference.op_solve_restart."""
    rng = np.random.default_rng(seed)
    m = D.shape[0]
    if forced is not None and 2 * D[0, int(forced)] > B + EPS:
        return 0.0, np.array([0], dtype=int), 0.0
    best_p, best_t, best_L = -1.0, None, None
    for shape in GAMMA_SET:
        t2, L2, unv2 = _fresh_start(D, prize, B, rng, shape=shape, forced=forced)
        if t2 is None:
            return 0.0, np.array([0], dtype=int), 0.0
        p2 = float(prize[t2].sum())
        if p2 > best_p + 1e-12 or (abs(p2 - best_p) <= 1e-12 and best_L is not None
                                   and L2 < best_L - 1e-12):
            best_p, best_t, best_L = p2, t2.copy(), L2
    t = best_t.copy()
    unv = np.ones(m, bool)
    unv[t] = False
    lo = 2 if forced is not None else 1
    for rd in range(ILS_ROUNDS):
        vis = t[lo:]
        if len(vis) == 0:
            break
        if rd % FRESH_EVERY == FRESH_EVERY - 1:
            t2, L2, unv2 = _fresh_start(D, prize, B, rng, forced=forced)
            if t2 is None:
                break
        else:
            mode = rd % 3
            if mode == 2:
                frac = rng.uniform(*DROP_FRAC_BIG)
                ndrop = min(max(1, int(round(frac * len(vis)))), len(vis))
                drop = rng.choice(vis, size=ndrop, replace=False)
            elif mode == 1:
                pred = np.roll(t, 1)
                succ = np.roll(t, -1)
                sav = D[pred, t] + D[t, succ] - D[pred, succ]
                ratio = prize[t] / (sav + 1e-9)
                ratio[0] = np.inf                  # depot never dropped
                if forced is not None and len(ratio) > 1:
                    ratio[1] = np.inf              # forced start never dropped
                noised = ratio * rng.uniform(0.6, 1.4, size=len(ratio))
                q = min(rng.integers(3, 6), len(vis))
                drop = t[np.argsort(noised)[:q]]
            else:
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
            t2, L2, unv2, _ = op_insert_loop(D, prize, t2, tour_len(D, t2), unv2, B,
                                             rng=rng, noise=REFILL_NOISE, forced=forced)
            t2, L2, unv2 = op_descend(D, prize, t2, unv2, B, forced=forced)
        _chk(t2, forced)
        p2 = float(prize[t2].sum())
        if p2 > best_p + 1e-12 or (abs(p2 - best_p) <= 1e-12 and L2 < best_L - 1e-12):
            best_p, best_t, best_L = p2, t2.copy(), L2
        if p2 >= best_p * (1.0 - 2.0 * WALK_TOL) - 1e-12:
            t, unv = t2, unv2
        else:
            t = best_t.copy()
            unv = np.ones(m, bool)
            unv[t] = False
    _chk(best_t, forced)
    return best_p, best_t, best_L


def _mk(n, inst_seed, alpha):
    rng = np.random.default_rng(inst_seed)
    depot = rng.uniform(size=2)
    loc = rng.uniform(size=(n, 2))
    xy = np.vstack([depot[None, :], loc])
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    d0 = D[0, 1:]
    prize = np.concatenate([[0.0], (1 + (d0 / d0.max() * 99).astype(int)) / 100.0])
    B = alpha * E.tour_len(D, E.nn_tour(D))
    return D, prize, B


def selftest_free_equivalence(seeds=(11, 22, 33), n=40, inst_seed=4242, alpha=0.5):
    """forced=None must reproduce E.op_solve_restart bit-for-bit."""
    D, prize, B = _mk(n, inst_seed, alpha)
    ok = True
    for sd in seeds:
        a = E.op_solve_restart(D, prize, B, sd)
        b = op_solve_restart(D, prize, B, sd, forced=None)
        teq = np.array_equal(np.asarray(a[1]), np.asarray(b[1]))
        same = abs(a[0] - b[0]) < 1e-12 and teq
        print("  seed %-4d original=%.6f locked_free=%.6f tours_equal=%s -> %s"
              % (sd, a[0], b[0], teq, "OK" if same else "DIVERGED"))
        ok = ok and same
    print("FREE_EQUIVALENCE %s" % ("OK" if ok else "FAILED"))
    return ok


def selftest_lock_holds(seeds=(11, 22, 33), n=40, inst_seed=4242, alpha=0.5):
    """the lock must hold, and forced must never beat free (structural bound)."""
    D, prize, B = _mk(n, inst_seed, alpha)
    free = max(op_solve_restart(D, prize, B, sd)[0] for sd in seeds)
    bad = 0
    checked = 0
    for s in range(1, n + 1):
        if 2 * D[0, s] > B:
            continue
        p, t, _ = op_solve_restart(D, prize, B, seeds[0], forced=s)
        checked += 1
        assert int(t[1]) == s, "LOCK BROKEN at s=%d" % s
        if p > free + 1e-9:
            bad += 1
    print("  lock held on %d feasible starts; forced>free violations = %d" % (checked, bad))
    print("LOCK_SELFTEST %s" % ("OK" if bad == 0 else "FAILED"))
    return bad == 0


if __name__ == "__main__":
    a = selftest_free_equivalence()
    b = selftest_lock_holds()
    print("ALL_SELFTESTS %s" % ("OK" if (a and b) else "FAILED"))
