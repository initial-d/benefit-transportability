"""Forge generators with dual exact solvers (public reimplementation, revision R1).

Five abstract reasoning families. Every generator returns a question, four
options, and a gold letter; generation aborts unless two algorithmically
distinct exact solvers agree.  Used for the prospectively locked Environment F
and for the stress environments of the revision.

    python scripts/forge_generate.py --env F --n-per-family 120 --seed 7101 --out data/forge/f_items.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from fractions import Fraction
from math import comb
from pathlib import Path

NAMES = ["Ankaa", "Bellatrix", "Capella", "Deneb", "Elnath", "Fomalhaut", "Gienah", "Hamal", "Izar", "Jabbah",
         "Kochab", "Lesath", "Mirach", "Nunki", "Okda", "Phact", "Rasalas", "Sabik", "Tarazed", "Unuk", "Vega",
         "Wezen", "Yildun", "Zaniah"]


class Abort(Exception):
    pass


def options_from(gold_value, distractors, fmt, rng):
    vals = [gold_value]
    for d in distractors:
        if d not in vals and len(vals) < 4:
            vals.append(d)
    if len(vals) < 4:
        raise Abort("not enough distractors")
    rng.shuffle(vals)
    letters = "ABCD"
    opts = [f"{letters[i]}. {fmt(v)}" for i, v in enumerate(vals)]
    return opts, letters[vals.index(gold_value)]


def near(v, rng, k=6, lo=1):
    cand = set()
    while len(cand) < k:
        d = rng.choice([-3, -2, -1, 1, 2, 3, 4, -4, 5])
        if v + d >= lo:
            cand.add(v + d)
    return list(cand)


# ----------------------------------------------------------------- temporal
def temporal(rng, tag, tier):
    """Critical path with resource release: tasks have durations; precedence DAG; answer: makespan."""
    n = {1: 9, 2: 11, 3: 13}[tier]
    dur = [rng.randint(2, 9) for _ in range(n)]
    preds = {i: set() for i in range(n)}
    for i in range(1, n):
        k = rng.randint(1, min(3, i))
        preds[i] = set(rng.sample(range(i), k))
    # solver 1: longest path by DP in topological order
    fin = [0] * n
    for i in range(n):
        fin[i] = max([fin[p] for p in preds[i]], default=0) + dur[i]
    s1 = max(fin)
    # solver 2: enumerate all source-to-node paths
    best = 0

    def walk(i, acc):
        nonlocal best
        acc += dur[i]
        succ = [j for j in range(n) if i in preds[j]]
        if not succ:
            best = max(best, acc)
        for j in succ:
            walk(j, acc)
    for i in range(n):
        if not preds[i]:
            walk(i, 0)
    if s1 != best:
        raise Abort("temporal solvers disagree")
    unit = "shifts"
    nm = lambda i: f"job {i + 1} of plan {tag}"  # noqa: E731
    lines = [f"{nm(i)} needs {dur[i]} {unit}" for i in range(n)]
    deps = [f"{nm(i)} may start only once {nm(p)} is complete" for i in range(n) for p in sorted(preds[i])]
    rng.shuffle(deps)
    q = (f"Work crews execute jobs in parallel whenever their prerequisites are complete. {'; '.join(lines)}. "
         f"Prerequisites: {'; '.join(deps)}. All jobs without prerequisites begin at time zero. "
         f"How many {unit} after the start is every job complete?")
    opts, gold = options_from(s1, near(s1, rng), lambda v: f"{v} {unit}", rng)
    feats = {"entities": n, "constraints": len(deps), "branching": max(len(p) for p in preds.values()), "depth": n, "precision": 0, "tier": tier}
    return q, opts, gold, feats


# ----------------------------------------------------------------- routing
def routing(rng, tag, tier):
    """Layered network, shortest path where exactly one edge may be traversed for free (voucher)."""
    layers = {1: 4, 2: 5, 3: 6}[tier]
    width = 3
    nodes = [(0, 0)] + [(l, k) for l in range(1, layers + 1) for k in range(width)] + [(layers + 1, 0)]
    idx = {v: i for i, v in enumerate(nodes)}
    edges = []
    for (l, k) in nodes:
        nxt = [v for v in nodes if v[0] == l + 1]
        for v in nxt:
            if rng.random() < 0.75 or len(nxt) == 1:
                edges.append((idx[(l, k)], idx[v], rng.randint(3, 19)))
    n = len(nodes)
    src, dst = 0, n - 1
    # solver 1: state-space Dijkstra over (node, voucher_used)
    import heapq
    INF = 10 ** 9
    dist = {(src, 0): 0}
    pq = [(0, src, 0)]
    adj = {i: [] for i in range(n)}
    for a, b, w in edges:
        adj[a].append((b, w))
    while pq:
        d, u, used = heapq.heappop(pq)
        if d > dist.get((u, used), INF):
            continue
        for v, w in adj[u]:
            for nu, cost in ([(used, w)] + ([(1, 0)] if not used else [])):
                nd = d + cost
                if nd < dist.get((v, nu), INF):
                    dist[(v, nu)] = nd
                    heapq.heappush(pq, (nd, v, nu))
    s1 = dist.get((dst, 1), INF)
    # solver 2: enumerate all paths, subtract the max edge
    best = INF

    def dfs(u, ws):
        nonlocal best
        if u == dst:
            best = min(best, sum(ws) - max(ws))
            return
        for v, w in adj[u]:
            dfs(v, ws + [w])
    dfs(src, [])
    if s1 != best or s1 >= INF:
        raise Abort("routing solvers disagree")
    lab = lambda i: "depot" if i == src else ("terminal" if i == dst else f"hub {i}")  # noqa: E731
    desc = [f"{lab(a)} to {lab(b)} costs {w}" for a, b, w in edges]
    rng.shuffle(desc)
    q = (f"Courier network {tag} has one-way lanes: {'; '.join(desc)}. A parcel travels from the depot to the terminal. "
         f"A single waiver lets exactly one lane on the chosen route be used free of charge. What is the lowest total cost?")
    opts, gold = options_from(s1, near(s1, rng, lo=0), lambda v: f"{v} fare units", rng)
    feats = {"entities": n, "constraints": len(edges), "branching": width, "depth": layers + 1, "precision": 0, "tier": tier}
    return q, opts, gold, feats


# ----------------------------------------------------------------- probability
def probability(rng, tag, tier):
    """Two dice-like sources; observe k sixes-like hits in m rolls with replacement; posterior of source A."""
    m = {1: 3, 2: 4, 3: 5}[tier]
    pa = Fraction(rng.randint(1, 5), 6)
    pb = Fraction(rng.randint(1, 5), 6)
    if pa == pb:
        raise Abort("equal")
    prior = Fraction(rng.choice([1, 1, 2, 1]), rng.choice([2, 3, 4, 5]))
    if prior >= 1:
        raise Abort("prior")
    k = rng.randint(1, m - 1)
    # solver 1: closed form
    la = comb(m, k) * pa ** k * (1 - pa) ** (m - k)
    lb = comb(m, k) * pb ** k * (1 - pb) ** (m - k)
    s1 = prior * la / (prior * la + (1 - prior) * lb)
    # solver 2: enumerate all outcome sequences
    num = den = Fraction(0)
    for src, pr, p in [("A", prior, pa), ("B", 1 - prior, pb)]:
        for seq in itertools.product([0, 1], repeat=m):
            if sum(seq) == k:
                w = pr
                for x in seq:
                    w *= p if x else (1 - p)
                den += w
                if src == "A":
                    num += w
    if num / den != s1:
        raise Abort("probability solvers disagree")
    ds = {prior, la / (la + lb), 1 - s1, Fraction(k, m), s1 + Fraction(1, 20), prior * la / (prior * la + lb)}
    ds.discard(s1)
    q = (f"Lab {tag} draws a sensor from batch A with probability {prior} and otherwise from batch B. Each test of a batch-A sensor "
         f"triggers with probability {pa}; each test of a batch-B sensor triggers with probability {pb}; tests are independent. "
         f"The drawn sensor is tested {m} times and triggers exactly {k} times. What is the posterior probability that it came from batch A?")
    opts, gold = options_from(s1, [d for d in ds if 0 < d < 1], lambda v: f"probability {v}", rng)
    feats = {"entities": 4, "constraints": 3, "branching": 2 ** m, "depth": m, "precision": 3, "tier": tier}
    return q, opts, gold, feats


# ----------------------------------------------------------------- assignment
def assignment(rng, tag, tier):
    """Seating with forbidden seats, adjacency and order constraints; unique solution required."""
    n = {1: 6, 2: 7, 3: 8}[tier]
    people = rng.sample(NAMES, n)
    target = list(range(n))
    rng.shuffle(target)
    pos = {people[i]: target[i] for i in range(n)}
    cons = []
    pool = []
    for p in people:
        bad = rng.choice([s for s in range(n) if s != pos[p]])
        pool.append(("not", p, bad))
    for a, b in itertools.permutations(people, 2):
        if pos[a] < pos[b]:
            pool.append(("before", a, b))
        if abs(pos[a] - pos[b]) == 1 and a < b:
            pool.append(("adj", a, b))
    rng.shuffle(pool)

    def ok(perm, cs):
        pp = {people[i]: perm[i] for i in range(n)}
        for c in cs:
            if c[0] == "not" and pp[c[1]] == c[2]:
                return False
            if c[0] == "before" and not pp[c[1]] < pp[c[2]]:
                return False
            if c[0] == "adj" and abs(pp[c[1]] - pp[c[2]]) != 1:
                return False
        return True
    sols = list(itertools.permutations(range(n)))
    for c in pool:
        cons.append(c)
        sols = [s for s in sols if ok(s, [c])]
        if len(sols) == 1:
            break
    if len(sols) != 1:
        raise Abort("not unique")
    slot = rng.randrange(n)
    s1 = [p for p in people if pos[p] == slot][0]
    # solver 2: backtracking from scratch
    found = []

    def bt(assign, used):
        if len(assign) == n:
            perm = tuple(assign[p] for p in people)
            if ok(perm, cons):
                found.append(dict(assign))
            return
        p = people[len(assign)]
        for s in range(n):
            if s not in used:
                assign[p] = s
                bt(assign, used | {s})
                del assign[p]
    bt({}, set())
    if len(found) != 1 or [p for p, s in found[0].items() if s == slot][0] != s1:
        raise Abort("assignment solvers disagree")
    full = lambda p: f"{p} of crew {tag}"  # noqa: E731
    txt = []
    for c in cons:
        if c[0] == "not":
            txt.append(f"{full(c[1])} is not in berth {c[2] + 1}")
        elif c[0] == "before":
            txt.append(f"{full(c[1])} is in a lower-numbered berth than {full(c[2])}")
        else:
            txt.append(f"{full(c[1])} and {full(c[2])} are in neighbouring berths")
    q = (f"{n} crew members occupy berths 1 through {n}, one per berth. {'; '.join(txt)}. Who is in berth {slot + 1}?")
    others = [p for p in people if p != s1]
    rng.shuffle(others)
    opts, gold = options_from(s1, others, lambda v: full(v), rng)
    feats = {"entities": n, "constraints": len(cons), "branching": n, "depth": n, "precision": 0, "tier": tier}
    return q, opts, gold, feats


# ----------------------------------------------------------------- optimization
def optimization(rng, tag, tier):
    """Two-constraint knapsack (weight and volume)."""
    n = {1: 9, 2: 11, 3: 13}[tier]
    items = [(rng.randint(2, 9), rng.randint(1, 6), rng.randint(4, 20)) for _ in range(n)]
    W = sum(i[0] for i in items) // 2
    V = sum(i[1] for i in items) // 2
    # solver 1: subset enumeration
    best = 0
    for mask in range(1 << n):
        w = v = val = 0
        for i in range(n):
            if mask >> i & 1:
                w += items[i][0]; v += items[i][1]; val += items[i][2]
        if w <= W and v <= V:
            best = max(best, val)
    # solver 2: 2-D dynamic programming
    dp = [[0] * (V + 1) for _ in range(W + 1)]
    for wi, vi, val in items:
        for w in range(W, wi - 1, -1):
            for v in range(V, vi - 1, -1):
                dp[w][v] = max(dp[w][v], dp[w - wi][v - vi] + val)
    if dp[W][V] != best:
        raise Abort("optimization solvers disagree")
    desc = [f"crate {i + 1} of manifest {tag} weighs {w} t, fills {v} m3 and is worth {val}" for i, (w, v, val) in enumerate(items)]
    q = (f"A barge can carry at most {W} t and {V} m3. {'; '.join(desc)}. Each crate is loaded at most once. "
         f"What is the largest total worth that fits within both limits?")
    opts, gold = options_from(best, near(best, rng), lambda v: f"worth {v}", rng)
    feats = {"entities": n, "constraints": 2 * n, "branching": n, "depth": n, "precision": 0, "tier": tier}
    return q, opts, gold, feats


GEN = {"temporal": temporal, "routing": routing, "probability": probability, "assignment": assignment, "optimization": optimization}


def generate(env: str, n_per_family: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for fam, fn in GEN.items():
        made = 0
        while made < n_per_family:
            tier = 1 + made * 3 // n_per_family
            tag = f"{env}{rng.randint(1000, 9999)}"
            try:
                q, opts, gold, feats = fn(rng, tag, tier)
            except Abort:
                continue
            out.append({"id": f"forge.{fam}.{env}.{made:03d}", "split": env, "family": fam, "question": q, "options": opts,
                        "gold": gold, "features": feats, "generator_partition": env})
            made += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--n-per-family", type=int, default=120)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = generate(a.env, a.n_per_family, a.seed)
    blob = json.dumps(items, ensure_ascii=False, indent=1)
    Path(a.out).write_text(blob, encoding="utf-8")
    print(a.out, len(items), "sha256", hashlib.sha256(blob.encode()).hexdigest())


if __name__ == "__main__":
    main()
