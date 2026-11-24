"""OPLib instance parser.

Parses OPLib .oplib files (TSPLIB-like format) from
https://github.com/bcamath-ds/OPLib and converts them to the Kool-convention
OP tuple used by our synthetic training distribution:
    (depot [2 floats], loc [n,2] in [0,1]^2, prize [n] in (0,1], max_length float)

Supported EDGE_WEIGHT_TYPE: EUC_2D, CEIL_2D.
GEO / EXPLICIT / ATT (and anything else) raise NotImplementedError; those
instances are not evaluated.

--------------------------------------------------------------------------
NORMALIZATION. Each choice listed explicitly:

 (a) Coordinates: translate so the bounding-box minimum is (0,0), then divide
     ALL coordinates by the LONGEST bounding-box side (single scalar scale).
     Preserves aspect ratio; coords land in [0,1] x [0,1] with at least one
     axis spanning the full [0,1].
 (b) Budget: COST_LIMIT divided by the same scalar scale. Caveats:
       - TSPLIB EUC_2D distances are rounded to the NEAREST INTEGER
         (nint(sqrt(...))). Continuous Euclidean distances after scaling are
         therefore an approximation even for EUC_2D (per-edge error <= 0.5
         original units; negligible for instances with large coordinates,
         non-negligible for tiny-coordinate instances).
       - CEIL_2D rounds UP; the rounding does NOT survive linear scaling, so
         the normalized budget for CEIL_2D instances is an approximation that
         is systematically optimistic (true integer-metric tours are longer
         than their continuous counterparts). Such instances are flagged via
         `ceil_2d_approx=True` in the normalized dict and must be reported.
 (c) Prizes: integer NODE_SCORE values divided by the MAX score over
     non-depot nodes, giving values in (0,1]. Note gen2/gen3/gen4 files
     assign a (nonzero) score to the depot node as well; the Kool convention
     has no depot prize, so the depot score is DROPPED (recorded in the
     normalized dict as `depot_score_dropped`).
 (d) Depot: the node listed in DEPOT_SECTION (always node 1 in OPLib) is
     moved to the Kool depot slot; the remaining nodes keep file order.
--------------------------------------------------------------------------

Pure numpy/stdlib. No torch.
"""

import math
import re
import sys
from dataclasses import dataclass, field

import numpy as np

SUPPORTED_EDGE_WEIGHT_TYPES = ("EUC_2D", "CEIL_2D")

_KEYWORD_RE = re.compile(r"^\s*([A-Z_0-9]+)\s*:\s*(.*?)\s*$")
_SECTION_NAMES = (
    "NODE_COORD_SECTION",
    "NODE_SCORE_SECTION",
    "DEPOT_SECTION",
    "EDGE_WEIGHT_SECTION",
    "DISPLAY_DATA_SECTION",
)


@dataclass
class RawOPInstance:
    """Instance as parsed from file: original units, integer scores."""
    name: str
    comment: str
    edge_weight_type: str
    dimension: int
    cost_limit: float
    coords: np.ndarray        # [dimension, 2] float64, file (node-id) order
    scores: np.ndarray        # [dimension] int64, file (node-id) order
    depot_index: int          # 0-based index into file order
    source_path: str = ""
    extra_keywords: dict = field(default_factory=dict)


def _tsplib_distance(a, b, edge_weight_type):
    """Integer TSPLIB distance between coordinate rows a and b (broadcasts)."""
    d = np.sqrt(((a - b) ** 2).sum(axis=-1))
    if edge_weight_type == "EUC_2D":
        return np.floor(d + 0.5).astype(np.int64)   # nint
    if edge_weight_type == "CEIL_2D":
        return np.ceil(d).astype(np.int64)
    raise NotImplementedError(edge_weight_type)


def parse_oplib(path):
    """Parse an .oplib file. Returns RawOPInstance.

    Raises NotImplementedError for unsupported EDGE_WEIGHT_TYPE and
    ValueError for malformed files.
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f]

    keywords = {}
    sections = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln or ln == "EOF":
            i += 1
            continue
        if ln in _SECTION_NAMES:
            sec = ln
            i += 1
            body = []
            while i < len(lines):
                nxt = lines[i]
                is_data = nxt and (nxt[0].isdigit() or nxt[0] == "-")
                if not nxt or nxt == "EOF" or nxt in _SECTION_NAMES or not is_data:
                    break
                body.append(nxt)
                i += 1
            sections[sec] = body
            continue
        m = _KEYWORD_RE.match(ln)
        if m:
            keywords[m.group(1)] = m.group(2)
            i += 1
            continue
        raise ValueError(f"{path}: unrecognized line {i + 1}: {ln!r}")

    ewt = keywords.get("EDGE_WEIGHT_TYPE", "").strip()
    if ewt not in SUPPORTED_EDGE_WEIGHT_TYPES:
        raise NotImplementedError(
            f"EDGE_WEIGHT_TYPE {ewt!r} not supported (file {path}); "
            f"only {SUPPORTED_EDGE_WEIGHT_TYPES} are implemented. "
            "GEO/EXPLICIT/ATT instances are deliberately excluded for now."
        )

    dimension = int(keywords["DIMENSION"])
    cost_limit = float(keywords["COST_LIMIT"])

    def _parse_id_rows(body, ncols, section):
        rows = np.array([ln.split() for ln in body], dtype=np.float64)
        if rows.shape != (len(body), 1 + ncols):
            raise ValueError(f"{path}: bad row shape in {section}")
        ids = rows[:, 0].astype(np.int64)
        if not np.array_equal(ids, np.arange(1, len(body) + 1)):
            raise ValueError(f"{path}: node ids in {section} not 1..n in order")
        return rows[:, 1:]

    coords = _parse_id_rows(sections["NODE_COORD_SECTION"], 2, "NODE_COORD_SECTION")
    scores_f = _parse_id_rows(sections["NODE_SCORE_SECTION"], 1, "NODE_SCORE_SECTION")[:, 0]
    scores = scores_f.astype(np.int64)
    if not np.allclose(scores, scores_f):
        raise ValueError(f"{path}: non-integer scores")

    depot_body = [ln for ln in sections["DEPOT_SECTION"] if ln != "-1"]
    if len(depot_body) != 1:
        raise ValueError(f"{path}: expected exactly one depot, got {depot_body}")
    depot_index = int(depot_body[0]) - 1
    if not (0 <= depot_index < dimension):
        raise ValueError(f"{path}: depot id out of range")

    if coords.shape[0] != dimension or scores.shape[0] != dimension:
        raise ValueError(
            f"{path}: DIMENSION={dimension} but {coords.shape[0]} coords / "
            f"{scores.shape[0]} scores"
        )

    consumed = {"NAME", "COMMENT", "TYPE", "DIMENSION", "COST_LIMIT", "EDGE_WEIGHT_TYPE"}
    extra = {k: v for k, v in keywords.items() if k not in consumed}

    return RawOPInstance(
        name=keywords.get("NAME", ""),
        comment=keywords.get("COMMENT", ""),
        edge_weight_type=ewt,
        dimension=dimension,
        cost_limit=cost_limit,
        coords=coords,
        scores=scores,
        depot_index=depot_index,
        source_path=str(path),
        extra_keywords=extra,
    )


def normalize(raw):
    """RawOPInstance -> Kool-convention dict (see module docstring).

    Returns dict with keys:
      depot [2], loc [n-1,2], prize [n-1], max_length float,
      scale (original units per normalized unit), ceil_2d_approx bool,
      depot_score_dropped int, name str.
    """
    lo = raw.coords.min(axis=0)
    span = raw.coords.max(axis=0) - lo
    scale = float(span.max())
    if scale <= 0:
        raise ValueError(f"{raw.name}: degenerate coordinates (zero bounding box)")
    norm_coords = (raw.coords - lo) / scale

    mask = np.ones(raw.dimension, dtype=bool)
    mask[raw.depot_index] = False
    non_depot_scores = raw.scores[mask]
    if (non_depot_scores <= 0).any():
        raise ValueError(f"{raw.name}: non-positive node score")
    prize = non_depot_scores / non_depot_scores.max()

    return {
        "name": raw.name,
        "depot": norm_coords[raw.depot_index].copy(),
        "loc": norm_coords[mask].copy(),
        "prize": prize.astype(np.float64),
        "max_length": raw.cost_limit / scale,
        "scale": scale,
        "ceil_2d_approx": raw.edge_weight_type == "CEIL_2D",
        "depot_score_dropped": int(raw.scores[raw.depot_index]),
    }


def feasibility_report(raw):
    """Sanity checks + reachability under the instance's OWN integer metric.

    A non-depot node i is reachable iff d(depot,i) + d(i,depot) <= COST_LIMIT
    with d the TSPLIB integer distance (nint for EUC_2D, ceil for CEIL_2D).
    """
    depot_xy = raw.coords[raw.depot_index]
    d = _tsplib_distance(raw.coords, depot_xy[None, :], raw.edge_weight_type)
    mask = np.ones(raw.dimension, dtype=bool)
    mask[raw.depot_index] = False
    reachable = (2 * d[mask]) <= raw.cost_limit
    return {
        "n_nodes": raw.dimension,
        "n_non_depot": int(mask.sum()),
        "n_reachable": int(reachable.sum()),
        "all_reachable": bool(reachable.all()),
        "checks": {
            "dimension_matches_coords": raw.coords.shape[0] == raw.dimension,
            "dimension_matches_scores": raw.scores.shape[0] == raw.dimension,
            "scores_nonneg": bool((raw.scores >= 0).all()),
            "non_depot_scores_positive": bool((raw.scores[mask] > 0).all()),
            "depot_valid": 0 <= raw.depot_index < raw.dimension,
            "coords_finite": bool(np.isfinite(raw.coords).all()),
            "cost_limit_positive": raw.cost_limit > 0,
        },
    }


if __name__ == "__main__":
    # Smoke test: parse every .oplib file in data/oplib/.
    import glob
    import os

    oplib_root = os.environ.get(
        "OPLIB_ROOT", os.path.join(os.environ.get("DATA_ROOT", "data"),
                                   "OPLib-master"))
    data_dir = os.path.join(oplib_root, "instances")
    files = sorted(glob.glob(os.path.join(data_dir, "gen*", "*.oplib")))
    if not files:
        print(f"no .oplib files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    n_ok = n_skip = 0
    for path in files:
        fname = os.path.basename(path)
        try:
            raw = parse_oplib(path)
        except NotImplementedError as e:
            print(f"[SKIP] {fname}: {e}")
            n_skip += 1
            continue
        norm = normalize(raw)
        rep = feasibility_report(raw)
        failed = [k for k, v in rep["checks"].items() if not v]
        print(f"[OK]   {fname}")
        print(f"       name={raw.name}  ewt={raw.edge_weight_type}  "
              f"dim={raw.dimension}  cost_limit={raw.cost_limit:g}  "
              f"depot_idx={raw.depot_index}")
        print(f"       scores: min={raw.scores.min()} max={raw.scores.max()} "
              f"depot_score_dropped={norm['depot_score_dropped']}  "
              f"prize: min={norm['prize'].min():.4f} max={norm['prize'].max():.4f}")
        print(f"       normalized: max_length={norm['max_length']:.4f}  "
              f"scale={norm['scale']:g}  ceil_2d_approx={norm['ceil_2d_approx']}  "
              f"loc in [{norm['loc'].min():.3f},{norm['loc'].max():.3f}]  "
              f"depot={np.round(norm['depot'], 3).tolist()}")
        print(f"       feasibility: {rep['n_reachable']}/{rep['n_non_depot']} "
              f"reachable  all_reachable={rep['all_reachable']}  "
              f"failed_checks={failed or 'none'}")
        assert not failed, f"{fname}: sanity checks failed: {failed}"
        assert norm["loc"].min() >= 0.0 and norm["loc"].max() <= 1.0
        assert 0.0 < norm["prize"].min() and norm["prize"].max() == 1.0
        n_ok += 1

    print(f"\nsmoke test: {n_ok} parsed, {n_skip} skipped "
          f"(unsupported edge weight type)")
    assert n_ok > 0
