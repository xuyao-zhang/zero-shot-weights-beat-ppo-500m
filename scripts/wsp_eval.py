# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Xuyao Zhang (comprehensive-weight-space-programming-leaderboard contributors)
"""Standalone Weight-Space Programming (WSP) MinAtar Experiment 5 parser,
evaluator, and dual-formula scoring library.

Implements:
- Exact ```edits block parser for 600 -> 32 -> 32 -> 6 tanh networks
- Official MinAtar==1.0.15 softmax evaluator (sticky_action_prob=0.1,
  difficulty_ramping=True, episodes=1000, seed=0, max_steps=10000)
- Writing Weights Index (WWI) and weight-space-bench (WSB) median formulas
- Weight-vector overlap/cosine comparison and two-sample episode-level z-test
"""

from __future__ import annotations

import ast
import logging
import math
import os
import re
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

GAMES = ("breakout", "seaquest", "asterix")

SHAPES = {
    "W1": (600, 32),
    "b1": (32,),
    "W2": (32, 32),
    "b2": (32,),
    "W3": (32, 6),
    "b3": (6,),
}

NPZ_KEYS = {
    "W1": "actor/dense1/kernel",
    "b1": "actor/dense1/bias",
    "W2": "actor/dense2/kernel",
    "b2": "actor/dense2/bias",
    "W3": "actor/logits/kernel",
    "b3": "actor/logits/bias",
}

MAX_ABS = 1e6
BLOCK = re.compile(r"```edits[^\n]*\n(.*?)(```|\Z)", re.S)
STMT = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*(\[(?P<idx>[^\]]*)\])?\s*(?P<op>[*+\-]?=)\s*(?P<val>.+?)\s*$",
    re.S,
)

CHANNELS = {
    "breakout": [["paddle"], [], ["brick"], ["ball"], ["trail"], []],
    "asterix": [["player"], ["enemy"], ["gold"], [], ["trail"], []],
    "seaquest": [
        ["sub_front", "sub_back"],
        ["enemy_fish", "enemy_sub", "enemy_bullet"],
        ["enemy_fish", "enemy_sub", "diver"],
        ["friendly_bullet"],
        ["trail", "sub_back"],
        ["oxygen_guage", "diver_guage"],
    ],
}

# Canonical MIT CSAIL paper baselines (MinAtar Experiment 5)
PPO500 = {"breakout": 25.344, "seaquest": 88.884, "asterix": 55.204}
PPO100 = {"breakout": 15.779, "seaquest": 5.617, "asterix": 25.820}
RANDOM_PAPER = {"breakout": 0.55, "seaquest": 0.09, "asterix": 0.49}
RANDOM_WSB_IMPLIED = {"breakout": 0.545, "seaquest": 0.093, "asterix": 0.492}
RANDOM_WIN_SEED0 = {"breakout": 0.539, "seaquest": 0.086, "asterix": 0.487}
RANDOM_ZERO = {"breakout": 0.0, "seaquest": 0.0, "asterix": 0.0}

# Aliases for explicit downstream imports
PPO500_BASELINES = PPO500
RANDOM_PAPER_BASELINES = RANDOM_PAPER


def zeros() -> dict[str, np.ndarray]:
    """Return an all-zero 600 -> 32 -> 32 -> 6 float32 weight dictionary."""
    return {k: np.zeros(s, np.float32) for k, s in SHAPES.items()}


def last_block(text: str) -> tuple[str | None, bool]:
    """Return (last ```edits block text or None, cut_off)."""
    blocks = list(BLOCK.finditer(text))
    if not blocks:
        return None, False
    m = blocks[-1]
    return m.group(1), m.group(2) != "```"


def statements(block: str) -> list[str]:
    """Split an edits block into statements, stripping '#' comments and joining multiline lists."""
    out: list[str] = []
    cur = ""
    depth = 0
    for line in block.split("\n"):
        line = line.split("#", 1)[0]
        if not line.strip() and depth == 0:
            continue
        cur = (cur + " " + line) if cur else line
        depth += line.count("[") - line.count("]")
        if depth <= 0:
            if cur.strip():
                out.append(cur.strip())
            cur, depth = "", 0
    if cur.strip():
        out.append(cur.strip())
    return out


def _index(s: str, ndim: int) -> tuple[int | slice, ...]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != ndim:
        raise ValueError(f"expected {ndim} indices")
    idx: list[int | slice] = []
    for p in parts:
        if ":" in p:
            bits = p.split(":")
            if len(bits) > 3:
                raise ValueError(f"bad range {p!r}")
            idx.append(slice(*[int(b) if b.strip() else None for b in bits]))
        else:
            idx.append(int(p))
    return tuple(idx)


def _value(s: str) -> np.ndarray:
    v = ast.literal_eval(s)
    a = np.array(v, dtype=np.float64)
    if a.dtype == object or not np.isfinite(a).all():
        raise ValueError("values must be finite numbers")
    if np.abs(a).max(initial=0) > MAX_ABS:
        raise ValueError(f"|value| > {MAX_ABS:g}")
    return a


def apply_edits(text: str, net: dict[str, np.ndarray] | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Parse the last ```edits block in text and apply statements to net (all-zero by default)."""
    net = net if net is not None else zeros()
    block, cut = last_block(text)
    rep: dict = {
        "block": block is not None,
        "statements": 0,
        "applied": 0,
        "entries_written": 0,
        "errors": [],
        "cut_off": cut,
    }
    for st in (statements(block) if block is not None else []):
        rep["statements"] += 1
        try:
            m = STMT.match(st)
            if not m:
                raise ValueError("not a statement")
            name, op = m.group(1), m.group("op")
            if name not in SHAPES:
                raise ValueError(f"unknown tensor {name!r}")
            t = net[name]
            idx = (
                _index(m.group("idx"), t.ndim)
                if m.group("idx") is not None
                else tuple(slice(None) for _ in t.shape)
            )
            target = t[idx]
            v = _value(m.group("val"))
            tshape = np.shape(target)
            if v.ndim and v.shape != tshape:
                raise ValueError(f"{v.size} values for a block of shape {tshape}")
            cur = np.asarray(target, np.float64)
            new = {
                "=": lambda: np.broadcast_to(v, tshape),
                "*=": lambda: cur * v,
                "+=": lambda: cur + v,
                "-=": lambda: cur - v,
            }[op]()
            new = np.asarray(new, np.float64)
            if np.abs(new).max(initial=0) > MAX_ABS:
                raise ValueError(f"result |value| > {MAX_ABS:g}")
            if tshape == ():
                t[idx] = np.float32(new)
            else:
                t[idx] = new.astype(np.float32)
            rep["applied"] += 1
            rep["entries_written"] += int(np.size(target))
        except Exception as e:  # noqa: BLE001
            rep["errors"].append({"statement": st[:200], "error": f"{type(e).__name__}: {e}"})
    rep["nonzero"] = {k: int(np.count_nonzero(v)) for k, v in net.items()}
    rep["nonzero_total"] = sum(rep["nonzero"].values())
    rep["max_abs"] = {k: float(np.abs(v).max()) for k, v in net.items()}
    return net, rep


apply = apply_edits


def save_npz(net: dict[str, np.ndarray], path: str | Path) -> None:
    """Save a weight dictionary using the canonical actor/* NPZ keys."""
    np.savez(path, **{NPZ_KEYS[k]: v for k, v in net.items()})


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a canonical actor/* NPZ file into float32 arrays."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[v].astype(np.float32) for k, v in NPZ_KEYS.items()}


def _env(game: str):
    logging.getLogger().setLevel(logging.ERROR)
    from minatar import Environment

    return Environment(game, sticky_action_prob=0.1, difficulty_ramping=True)


def _episodes(args: tuple[str, dict[str, np.ndarray], list[int], int]) -> list[tuple[float, int]]:
    game, net, seeds, max_steps = args
    env = _env(game)
    src = [[env.env.channels[n] for n in names] for names in CHANNELS[game]]
    W1, b1, W2, b2, W3, b3 = (
        net[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2", "W3", "b3")
    )
    out: list[tuple[float, int]] = []
    for s in seeds:
        env.seed(int(s))
        env.reset()
        rng = np.random.default_rng([1, int(s)])
        total, steps, done = 0.0, 0, False
        while not done and steps < max_steps:
            st = env.state()
            x = np.zeros((10, 10, 6), bool)
            for ch, idx in enumerate(src):
                if idx:
                    x[:, :, ch] = st[:, :, idx].any(-1)
            nz = np.flatnonzero(x.reshape(-1))
            h1 = np.tanh(W1[nz].sum(0) + b1)
            h2 = np.tanh(h1 @ W2 + b2)
            z = h2 @ W3 + b3
            p = np.exp(z - z.max())
            a = int(rng.choice(6, p=p / p.sum()))
            r, done = env.act(a)
            total += r
            steps += 1
        out.append((total, steps))
    return out


def score_network(
    net: dict[str, np.ndarray],
    game: str,
    episodes: int = 1000,
    seed: int = 0,
    max_steps: int = 10000,
    workers: int = 0,
) -> dict[str, float | int]:
    """Evaluate a 600->32->32->6 tanh policy network over MinAtar episodes."""
    seeds = list(range(seed, seed + episodes))
    resolved_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    chunks = [seeds[i::resolved_workers] for i in range(resolved_workers) if seeds[i::resolved_workers]]
    net_arrays = {k: np.asarray(v) for k, v in net.items()}
    res: list[tuple[float, int]] = []
    if len(chunks) == 1:
        res = _episodes((game, net_arrays, chunks[0], max_steps))
    else:
        with ProcessPoolExecutor(len(chunks)) as ex:
            for part in ex.map(_episodes, [(game, net_arrays, c, max_steps) for c in chunks]):
                res += part
    r = np.array([x[0] for x in res], dtype=np.float64)
    n = np.array([x[1] for x in res], dtype=np.float64)
    return {
        "mean": float(r.mean()),
        "se": float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0,
        "episodes": int(len(r)),
        "max": float(r.max()),
        "mean_length": float(n.mean()),
    }


score = score_network


def _extract_score(val: float | int | dict[str, float]) -> float:
    if isinstance(val, dict):
        return float(val["mean"])
    return float(val)


def wwi_game_score(score_val: float, rand_val: float, ppo500_val: float) -> float:
    """Writing Weights Index per-game score: 100 * q / (1 + q)."""
    q = max(0.0, float(score_val) - float(rand_val)) / (float(ppo500_val) - float(rand_val))
    return 100.0 * q / (1.0 + q)


def wwi_overall(
    scores_dict: dict[str, float | dict[str, float]],
    random_dict: dict[str, float] = RANDOM_PAPER,
    ppo500_dict: dict[str, float] = PPO500,
) -> float:
    """Overall Writing Weights Index (arithmetic mean across Breakout, Seaquest, Asterix)."""
    per_game = [
        wwi_game_score(_extract_score(scores_dict[g]), random_dict[g], ppo500_dict[g])
        for g in GAMES
    ]
    return sum(per_game) / len(GAMES)


compute_wwi = wwi_overall


def wsb_game_norm(score_val: float, rand_val: float, ppo500_val: float) -> float:
    """weight-space-bench per-game normalized score (fraction of PPO-500M above Random)."""
    return (float(score_val) - float(rand_val)) / (float(ppo500_val) - float(rand_val))


def wsb_median_score(
    scores_dict: dict[str, float | dict[str, float]],
    random_dict: dict[str, float] = RANDOM_PAPER,
    ppo500_dict: dict[str, float] = PPO500,
) -> float:
    """weight-space-bench overall score: 100 * median normalized score across the 3 games."""
    norms = [
        wsb_game_norm(_extract_score(scores_dict[g]), random_dict[g], ppo500_dict[g])
        for g in GAMES
    ]
    return 100.0 * float(statistics.median(norms))


compute_wsb_median = wsb_median_score


def compare_networks(net_a: dict[str, np.ndarray], net_b: dict[str, np.ndarray]) -> dict[str, float | int]:
    """Compare two 600->32->32->6 networks for nonzero overlap and full-vector cosine similarity."""
    flat_a = np.concatenate([np.asarray(net_a[k], dtype=np.float64).ravel() for k in SHAPES])
    flat_b = np.concatenate([np.asarray(net_b[k], dtype=np.float64).ravel() for k in SHAPES])
    nz_a = flat_a != 0.0
    nz_b = flat_b != 0.0
    overlap = int(np.sum(nz_a & nz_b))
    union = int(np.sum(nz_a | nz_b))
    identical = int(np.sum(nz_a & nz_b & (flat_a == flat_b)))
    norm_a = float(np.linalg.norm(flat_a))
    norm_b = float(np.linalg.norm(flat_b))
    cosine = float(np.dot(flat_a, flat_b) / (norm_a * norm_b)) if (norm_a > 0.0 and norm_b > 0.0) else 0.0
    return {
        "nonzero_a": int(np.sum(nz_a)),
        "nonzero_b": int(np.sum(nz_b)),
        "overlapping_nonzero_coords": overlap,
        "union_nonzero_coords": union,
        "jaccard": float(overlap / union) if union > 0 else 0.0,
        "identical_nonzero_coords": identical,
        "cosine_similarity": cosine,
    }


def two_sample_z_test(
    mean_a: float,
    se_a: float,
    mean_b: float,
    se_b: float,
) -> dict[str, float]:
    """Two-sample z-test between two independent 1,000-episode evaluation means."""
    delta = float(mean_b) - float(mean_a)
    pct_gain = (delta / float(mean_a)) * 100.0 if mean_a != 0 else 0.0
    se_diff = math.sqrt(float(se_a) ** 2 + float(se_b) ** 2)
    z_stat = delta / se_diff
    p_value = math.erfc(abs(z_stat) / math.sqrt(2.0))
    return {
        "delta_score": delta,
        "pct_gain": pct_gain,
        "se_diff": se_diff,
        "z_stat": z_stat,
        "p_value_two_sided": p_value,
    }


if __name__ == "__main__":
    print(
        f"wsp_eval ready | PPO500 WWI={wwi_overall(PPO500):.4f}, "
        f"PPO500 WSB={wsb_median_score(PPO500):.4f}, "
        f"PPO100 WWI={wwi_overall(PPO100):.4f}"
    )
