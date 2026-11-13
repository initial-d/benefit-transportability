"""Revision analyses for NEUCOM-D-26-21475 (R1).

Offline re-analysis of the released paired outcomes; no model calls are made.

Usage:
    python -I revision_analyses.py <release_dir> <out_dir>

<release_dir> is a checkout of the public benefit-transportability repository
(it must contain results/primary, results/qwen_control and results/livebench).

Note on environment C (primary pair): the public release contains family-level
benefits for C but not its item rows.  C enters only as a historical
environment, through its family means (60 items per family), which is exactly
how the manuscript uses it.  For interval estimates we reconstruct C counts
from the released family benefits assuming no fast-only wins; fast-only wins
occur in 4 of 1,600 released primary-pair items, all in A.
"""
from __future__ import annotations

import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

FAM = ["temporal", "routing", "probability", "assignment", "optimization"]
GAMMA = 0.15
SEED = 20261113
C_FAMILY_BENEFIT = {"temporal": 5, "routing": 3, "probability": 10, "assignment": 10, "optimization": 13}  # counts of +1 out of 60
FEATS = ["entities", "constraints", "branching", "depth", "precision", "tier"]


# --------------------------------------------------------------------------- data
class Pair:
    """Paired outcomes for one configuration pair: arrays over items."""

    def __init__(self, name, env, fam, X, yl, yu, kl, ku, ids):
        self.name = name
        self.env = np.asarray(env)
        self.fam = np.asarray(fam)
        self.X = np.asarray(X, dtype=float)
        self.yl = np.asarray(yl, dtype=float)
        self.yu = np.asarray(yu, dtype=float)
        self.kl = np.asarray(kl, dtype=float)
        self.ku = np.asarray(ku, dtype=float)
        self.ids = np.asarray(ids)

    @property
    def b(self):
        return self.yu - self.yl

    def sub(self, mask):
        return Pair(self.name, self.env[mask], self.fam[mask], self.X[mask], self.yl[mask], self.yu[mask],
                    self.kl[mask], self.ku[mask], self.ids[mask])

    def envs(self, *names):
        return self.sub(np.isin(self.env, names))


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def tok(r):
    return float((r.get("usage") or {}).get("total_tokens") or 0)


def feat(r):
    f = r.get("features") or {}
    return [float(f.get(k, 0) or 0) for k in FEATS]


def load_main(rel: Path) -> Pair:
    files = [
        ("primary/forge_abc_query_metrics.jsonl", {"calibration": "A", "validation": "B"}),
        ("primary/locked_d_query_metrics.jsonl", {"test": "D"}),
        ("primary/target_audit_e_calibration_metrics.jsonl", {"calibration": "E_audit"}),
        ("primary/target_audit_e_test_metrics.jsonl", {"test": "E_test"}),
    ]
    units = collections.defaultdict(dict)
    for f, smap in files:
        for r in read_jsonl(rel / "results" / f):
            if r.get("error"):
                continue
            units[(smap[r["split"]], r["id"])][r["mode"]] = r
    rows = []
    for (env, iid), u in units.items():
        if {"fast", "slow"} <= u.keys():
            f, s = u["fast"], u["slow"]
            rows.append((env, f["family"], feat(f), int(f["correct"]), int(s["correct"]), tok(f), tok(s), iid))
    # pseudo C (family means only)
    for fam in FAM:
        k = C_FAMILY_BENEFIT[fam]
        for i in range(60):
            rows.append(("C", fam, [np.nan] * 6, 0 if i < k else 1, 1, np.nan, np.nan, f"C.{fam}.{i}"))
    env, fam, X, yl, yu, kl, ku, ids = zip(*rows)
    return Pair("DeepSeek-V3.2 -> Qwen3.5-397B", env, fam, X, yl, yu, kl, ku, ids)


def load_qwen(rel: Path, lower="small", upper="large") -> Pair:
    latest = {}
    for r in read_jsonl(rel / "results/qwen_control/same_series_qwen_query_metrics.jsonl"):
        if not r.get("error"):
            latest[r["key"]] = r
    units = collections.defaultdict(dict)
    for r in latest.values():
        units[r["id"]][r["tier"]] = r
    rows = []
    for iid, u in units.items():
        if {lower, upper} <= u.keys():
            a, b = u[lower], u[upper]
            rows.append((a["source_env"], a["family"], feat(a), int(a["correct"]), int(b["correct"]), tok(a), tok(b), iid))
    env, fam, X, yl, yu, kl, ku, ids = zip(*rows)
    names = {"small": "Qwen3.5-35B", "mid": "Qwen3.5-122B", "large": "Qwen3.5-397B"}
    return Pair(f"{names[lower]} -> {names[upper]}", env, fam, X, yl, yu, kl, ku, ids)


def qwen_tier_table(rel: Path):
    latest = {}
    for r in read_jsonl(rel / "results/qwen_control/same_series_qwen_query_metrics.jsonl"):
        if not r.get("error"):
            latest[r["key"]] = r
    out = collections.defaultdict(dict)
    for r in latest.values():
        out[r["id"]][r["tier"]] = r
    return out


def load_livebench(rel: Path) -> Pair:
    units = collections.defaultdict(dict)
    for r in read_jsonl(rel / "results/livebench/scored_query_metrics.jsonl"):
        if not r.get("error"):
            units[r["id"]][r["tier"]] = r
    rows = []
    for iid, u in units.items():
        if {"small", "large"} <= u.keys():
            a, b = u["small"], u["large"]
            rows.append((a["livebench_release_date"], a["category"], [0.0] * 6, float(a["score"]), float(b["score"]),
                         tok(a), tok(b), iid + "|" + a["task"]))
    env, fam, X, yl, yu, kl, ku, ids = zip(*rows)
    return Pair("Qwen3.5-35B -> Qwen3.5-397B (LiveBench)", env, fam, X, yl, yu, kl, ku, ids)


# ---------------------------------------------------------------- small helpers
def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k + 1)]
    m = max(logs)
    return min(1.0, math.exp(m) * sum(math.exp(x - m) for x in logs))


def clopper_pearson(k, n, alpha=0.05):
    def solve(target, kk):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if binom_cdf(kk, n, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(1 - alpha / 2, k - 1)
    upper = 1.0 if k == n else solve(alpha / 2, k)
    return lower, upper


def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=float), q))


def spearman(a, b):
    def rank(v):
        v = np.asarray(v, dtype=float)
        order = v.argsort()
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for val in np.unique(v):  # average ties
            m = v == val
            r[m] = r[m].mean()
        return r
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def fam_tau(p: Pair, fams=FAM):
    return np.array([p.b[p.fam == f].mean() if (p.fam == f).any() else np.nan for f in fams])


def fam_dk(p: Pair, fams=FAM):
    return np.array([np.nanmean((p.ku - p.kl)[p.fam == f]) if (p.fam == f).any() else 0.0 for f in fams])


def w1_discrete(b1, b2):
    """Wasserstein-1 distance between two empirical distributions on the real line."""
    xs = np.sort(np.unique(np.concatenate([b1, b2])))
    if len(xs) < 2:
        return 0.0
    F1 = np.array([(b1 <= x).mean() for x in xs[:-1]])
    F2 = np.array([(b2 <= x).mean() for x in xs[:-1]])
    return float(np.sum(np.abs(F1 - F2) * np.diff(xs)))


# --------------------------------------------------------- 1. shift quantification
def shift_quantification(p: Pair, envs, fams, rng, B=2000, P=2000):
    res = {"pair": p.name, "envs": envs, "families": fams, "cells": [], "pairs": [], "homogeneity": []}
    idx = {(e, f): np.where((p.env == e) & (p.fam == f))[0] for e in envs for f in fams}
    b = p.b
    # cell CIs
    for e in envs:
        for f in fams:
            ii = idx[(e, f)]
            if len(ii) == 0:
                continue
            bb = b[ii]
            boots = bb[rng.integers(0, len(bb), size=(B, len(bb)))].mean(axis=1)
            res["cells"].append({
                "env": e, "family": f, "n": int(len(bb)), "tau": float(bb.mean()),
                "lo": pct(boots, 2.5), "hi": pct(boots, 97.5),
                "n_pos": int((bb > 0).sum()), "n_zero": int((bb == 0).sum()), "n_neg": int((bb < 0).sum()),
                "mean_lower": float(p.yl[ii].mean()), "mean_upper": float(p.yu[ii].mean()),
            })

    def stats_for(sample_idx):
        taus = {e: np.array([b[sample_idx[(e, f)]].mean() if len(sample_idx[(e, f)]) else np.nan for f in fams]) for e in envs}
        out = {}
        for i, e in enumerate(envs):
            for e2 in envs[i + 1:]:
                m = ~np.isnan(taus[e]) & ~np.isnan(taus[e2])
                if not m.any():
                    continue
                d = float(np.mean(np.abs(taus[e][m] - taus[e2][m])))
                w = float(np.mean([w1_discrete(b[sample_idx[(e, f)]], b[sample_idx[(e2, f)]]) for f, mm in zip(fams, m) if mm]))
                rho = spearman(taus[e][m], taus[e2][m]) if m.sum() >= 3 else float("nan")
                out[(e, e2)] = (d, w, rho)
        return out

    obs = stats_for(idx)
    boot = collections.defaultdict(list)
    for _ in range(B):
        sidx = {k: v[rng.integers(0, len(v), size=len(v))] if len(v) else v for k, v in idx.items()}
        for k, v in stats_for(sidx).items():
            boot[k].append(v)
    # permutation null for each env pair: shuffle env labels within family
    for (e, e2), (d, w, rho) in obs.items():
        bd = np.array(boot[(e, e2)])
        null_d, null_w = [], []
        for _ in range(P):
            ds, ws = [], []
            for f in fams:
                i1, i2 = idx[(e, f)], idx[(e2, f)]
                if len(i1) == 0 or len(i2) == 0:
                    continue
                pool = np.concatenate([b[i1], b[i2]])
                rng.shuffle(pool)
                x1, x2 = pool[:len(i1)], pool[len(i1):]
                ds.append(abs(x1.mean() - x2.mean()))
                ws.append(w1_discrete(x1, x2))
            null_d.append(np.mean(ds))
            null_w.append(np.mean(ws))
        null_d, null_w = np.array(null_d), np.array(null_w)
        res["pairs"].append({
            "env1": e, "env2": e2,
            "profile_l1": d, "profile_l1_lo": pct(bd[:, 0], 2.5), "profile_l1_hi": pct(bd[:, 0], 97.5),
            "profile_l1_null_mean": float(null_d.mean()), "profile_l1_perm_p": float((1 + (null_d >= d - 1e-12).sum()) / (P + 1)),
            "w1": w, "w1_lo": pct(bd[:, 1], 2.5), "w1_hi": pct(bd[:, 1], 97.5),
            "w1_null_mean": float(null_w.mean()), "w1_perm_p": float((1 + (null_w >= w - 1e-12).sum()) / (P + 1)),
            "spearman": rho,
            "spearman_lo": pct(bd[~np.isnan(bd[:, 2]), 2], 2.5) if np.isfinite(bd[:, 2]).any() else float("nan"),
            "spearman_hi": pct(bd[~np.isnan(bd[:, 2]), 2], 97.5) if np.isfinite(bd[:, 2]).any() else float("nan"),
        })
    # summary over env pairs with bootstrap CI
    keys = list(obs.keys())
    if not keys:
        return res
    bmat = np.array([[boot[k][i] for k in keys] for i in range(B)])  # B x pairs x 3
    summ = {}
    for j, name in enumerate(["profile_l1", "w1", "spearman"]):
        vals = np.array([obs[k][j] for k in keys])
        bvals = np.nanmean(bmat[:, :, j], axis=1)
        summ[name] = {"mean": float(np.nanmean(vals)), "lo": pct(bvals, 2.5), "hi": pct(bvals, 97.5),
                      "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))}
    summ["profile_l1"]["null_mean"] = float(np.mean([r["profile_l1_null_mean"] for r in res["pairs"]]))
    summ["w1"]["null_mean"] = float(np.mean([r["w1_null_mean"] for r in res["pairs"]]))
    summ["n_pairs_sig_l1"] = int(sum(r["profile_l1_perm_p"] < 0.05 for r in res["pairs"]))
    summ["n_pairs"] = len(res["pairs"])
    res["summary"] = summ
    # per-family homogeneity across all envs (weighted between-env variance), permutation p
    for f in fams:
        groups = [idx[(e, f)] for e in envs if len(idx[(e, f)])]
        if len(groups) < 2:
            continue
        vals = [b[g] for g in groups]
        allv = np.concatenate(vals)

        def stat(vs):
            mu = np.concatenate(vs).mean()
            return sum(len(v) * (v.mean() - mu) ** 2 for v in vs)
        s_obs = stat(vals)
        sizes = np.cumsum([len(v) for v in vals])[:-1]
        cnt = 0
        for _ in range(P):
            perm = rng.permutation(allv)
            if stat(np.split(perm, sizes)) >= s_obs - 1e-12:
                cnt += 1
        taus = [v.mean() for v in vals]
        res["homogeneity"].append({"family": f, "range": float(max(taus) - min(taus)), "stat": float(s_obs),
                                   "perm_p": float((cnt + 1) / (P + 1))})
    return res


# ------------------------------------------------------- 2. policies (family level)
def one_fast(est, dk, gamma=GAMMA, k=1, eligible_mask=None):
    """Return boolean fast-mask over FAM.  Leave up to k eligible families fast,
    preferring larger incremental token cost (the frozen E rule)."""
    est = np.asarray(est, dtype=float)
    elig = [i for i in range(len(est)) if np.isfinite(est[i]) and est[i] <= gamma + 1e-12
            and (eligible_mask is None or eligible_mask[i])]
    elig.sort(key=lambda i: (-dk[i], FAM[i] if i < len(FAM) else ""))
    fast = np.zeros(len(est), dtype=bool)
    for i in elig[:k]:
        fast[i] = True
    return fast


def threshold_fast(est, gamma=GAMMA):
    est = np.asarray(est, dtype=float)
    return np.isfinite(est) & (est <= gamma + 1e-12)


def evaluate_family(p: Pair, fast_mask, fams=FAM):
    fm = dict(zip(fams, fast_mask))
    is_fast = np.array([fm.get(f, False) for f in p.fam])
    return evaluate_items(p, is_fast)


def evaluate_items(p: Pair, is_fast):
    y = np.where(is_fast, p.yl, p.yu)
    k = np.where(is_fast, p.kl, p.ku)
    ff = int(((p.yu > p.yl) & is_fast).sum())
    loss = float(np.clip(p.yu - p.yl, 0, None)[is_fast].sum())
    return {"n": int(len(y)), "acc": float(y.mean()), "tokens": float(k.mean()), "slow_rate": float(1 - is_fast.mean()),
            "ff": ff, "ff_loss": loss, "save_vs_slow": float(1 - k.sum() / p.ku.sum()) if np.isfinite(p.ku).all() else float("nan")}


def random_matched(p: Pair, slow_rate):
    """Exact expectation of slow-rate-matched random routing."""
    acc = slow_rate * p.yu.mean() + (1 - slow_rate) * p.yl.mean()
    tokens = slow_rate * p.ku.mean() + (1 - slow_rate) * p.kl.mean()
    ff = (1 - slow_rate) * (p.yu > p.yl).sum()
    return {"acc": float(acc), "tokens": float(tokens), "ff": float(ff), "slow_rate": float(slow_rate)}


def ucb_net(bvals, z=1.645):
    """One-sided upper bound for a mean paired benefit in [-1,1] (Wald with Agresti-style +2 smoothing)."""
    n = len(bvals)
    if n == 0:
        return 1.0
    pos = (bvals > 0).sum() + 1
    neg = (bvals < 0).sum() + 1
    m = n + 2
    mu = (pos - neg) / m
    var = (pos + neg) / m - mu ** 2
    return float(mu + z * math.sqrt(max(var, 1e-12) / m))


def hoeffding_ucb(bvals, eta=0.05):
    n = len(bvals)
    return float(np.mean(bvals) + math.sqrt(2 * math.log(1 / eta) / n)) if n else 1.0


def posterior_prob_le(bvals, gamma=GAMMA, draws=20000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    pos, zero, neg = (bvals > 0).sum(), (bvals == 0).sum(), (bvals < 0).sum()
    d = rng.dirichlet([pos + 0.5, zero + 0.5, neg + 0.5], size=draws)
    return float(((d[:, 0] - d[:, 2]) <= gamma).mean())


# ------------------------------------------------------------- item-level routers
def standardize(Xtr, Xte):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def design(p: Pair):
    oh = np.array([[1.0 if f == g else 0.0 for g in FAM] for f in p.fam])
    return np.hstack([p.X, oh])


def logistic_fit(X, y, lam=1.0, iters=50):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X1.shape[1])
    R = lam * np.eye(X1.shape[1])
    R[0, 0] = 0
    for _ in range(iters):
        z = X1 @ w
        pr = 1 / (1 + np.exp(-z))
        g = X1.T @ (pr - y) + R @ w
        H = X1.T @ (X1 * (pr * (1 - pr))[:, None]) + R + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def logistic_pred(w, X):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    return 1 / (1 + np.exp(-(X1 @ w)))


def ridge_fit(X, y, lam=1.0):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    R = lam * np.eye(X1.shape[1])
    R[0, 0] = 0
    return np.linalg.solve(X1.T @ X1 + R, X1.T @ y)


def ridge_pred(w, X):
    return np.hstack([np.ones((len(X), 1)), X]) @ w


def budget_fast(score_fast_priority, slow_rate):
    """Send the (1-slow_rate) fraction with the highest fast-priority to fast."""
    n = len(score_fast_priority)
    k = int(round((1 - slow_rate) * n))
    order = np.argsort(-score_fast_priority, kind="stable")
    m = np.zeros(n, dtype=bool)
    m[order[:k]] = True
    return m


def item_routers(audit: Pair, test: Pair, slow_rate, rng):
    """Learned routers calibrated on the same paired target audit."""
    Xa, Xt = standardize(design(audit), design(test))
    out = {}
    rescue = ((audit.yu > audit.yl)).astype(float)
    w = logistic_fit(Xa, rescue, lam=1.0)
    pr = logistic_pred(w, Xt)
    out["Learned rescue classifier (RouteLLM-style)"] = evaluate_items(test, budget_fast(-pr, slow_rate))
    wd = ridge_fit(Xa, audit.ku - audit.kl, lam=1.0)
    dk = np.maximum(ridge_pred(wd, Xt), 1.0)
    out["Cost-sensitive knapsack (rescue per token)"] = evaluate_items(test, budget_fast(dk / np.maximum(pr, 0.02), slow_rate))
    wnet = ridge_fit(Xa, audit.b, lam=1.0)
    out["Net-benefit regression"] = evaluate_items(test, budget_fast(-ridge_pred(wnet, Xt), slow_rate))
    # uncertainty-aware: bootstrap ensemble, rank by mean + sd of rescue probability
    preds = []
    for _ in range(30):
        ii = rng.integers(0, len(Xa), size=len(Xa))
        if rescue[ii].min() == rescue[ii].max():
            continue
        preds.append(logistic_pred(logistic_fit(Xa[ii], rescue[ii], lam=1.0), Xt))
    P = np.array(preds)
    out["Uncertainty-aware ensemble (mean+sd)"] = evaluate_items(test, budget_fast(-(P.mean(0) + P.std(0)), slow_rate))
    # difficulty / confidence surrogate: predicted fast correctness
    wf = logistic_fit(Xa, audit.yl, lam=1.0)
    out["Fast-correctness (difficulty) router"] = evaluate_items(test, budget_fast(logistic_pred(wf, Xt), slow_rate))
    return out


# ----------------------------------------------------------- family policy suite
def family_policy_suite(hist: Pair, audit: Pair, test: Pair, rng, gamma=GAMMA, k_fast=1, slow_rate=0.8,
                        hist_envs=None, with_items=True, delta0=0.025):
    hist_tau = fam_tau(hist)
    audit_tau = fam_tau(audit)
    dk = np.maximum(fam_dk(audit), 0)
    res = {}
    res["Historical point estimate"] = evaluate_family(test, one_fast(hist_tau, dk, gamma, k_fast))
    # historical robust: smallest worst-observed benefit across historical envs
    envs = hist_envs or sorted(set(hist.env))
    worst = np.array([max(hist.b[(hist.env == e) & (hist.fam == f)].mean() for e in envs if ((hist.env == e) & (hist.fam == f)).any()) for f in FAM])
    rmask = np.zeros(len(FAM), dtype=bool)
    rmask[int(np.argmin(worst))] = True
    res["Historical robust (worst observed)"] = evaluate_family(test, rmask)
    res["Target audit, point estimate (ours)"] = evaluate_family(test, one_fast(audit_tau, dk, gamma, k_fast))
    ucbs = np.array([ucb_net(audit.b[audit.fam == f]) for f in FAM])
    um = np.zeros(len(FAM), dtype=bool)
    um[int(np.argmin(ucbs))] = True
    res["Target audit, min upper bound"] = evaluate_family(test, um)
    hoef = np.array([hoeffding_ucb(audit.b[audit.fam == f]) for f in FAM])
    res["Target audit, certified (Hoeffding UCB<=gamma)"] = evaluate_family(test, one_fast(hoef, dk, gamma, k_fast))
    post = np.array([posterior_prob_le(audit.b[audit.fam == f], gamma, 4000, rng) for f in FAM])
    pm = np.zeros(len(FAM), dtype=bool)
    pm[int(np.argmax(post))] = True
    res["Target audit, Bayesian P(tau<=gamma)"] = evaluate_family(test, pm)
    # empirical-Bayes shrinkage toward historical mean, prior weight n0 = 20
    n_a = np.array([(audit.fam == f).sum() for f in FAM])
    shr = (n_a * audit_tau + 20 * hist_tau) / (n_a + 20)
    res["Historical prior + target audit (shrinkage)"] = evaluate_family(test, one_fast(shr, dk, gamma, k_fast))
    # focused boundary-and-incumbent audit
    inc = one_fast(hist_tau, dk, gamma, k_fast)
    margin = np.abs(hist_tau - gamma)
    audited = inc | (margin <= delta0 + 1e-12)
    mixed = np.where(audited, audit_tau, hist_tau)
    r = evaluate_family(test, one_fast(mixed, dk, gamma, k_fast))
    r["audit_calls"] = int(sum(n_a[audited]))
    res["Focused audit (boundary+incumbent)"] = r
    if with_items and np.isfinite(audit.X).all():
        res.update(item_routers(audit, test, slow_rate, rng))
    rm = random_matched(test, slow_rate)
    res["Random @ matched slow rate"] = {**rm, "save_vs_slow": float(1 - rm["tokens"] / test.ku.mean())}
    # hindsight oracle one-fast family on test (not deployable)
    best = None
    for i in range(len(FAM)):
        m = np.zeros(len(FAM), dtype=bool)
        m[i] = True
        e = evaluate_family(test, m)
        if best is None or (e["acc"], -e["tokens"]) > (best["acc"], -best["tokens"]):
            best = e
    res["Hindsight best single fast family"] = best
    res["Uniform slow"] = evaluate_items(test, np.zeros(len(test.yl), dtype=bool))
    return res


def split_balanced(p: Pair, per_family, rng):
    a_idx, t_idx = [], []
    for f in FAM:
        ii = np.where(p.fam == f)[0]
        ii = ii[rng.permutation(len(ii))]
        a_idx.extend(ii[:per_family])
        t_idx.extend(ii[per_family:])
    a = np.zeros(len(p.fam), dtype=bool)
    a[a_idx] = True
    return p.sub(a), p.sub(~a)


def aggregate_runs(runs):
    keys = runs[0].keys()
    out = {}
    for k in keys:
        acc = np.array([r[k]["acc"] for r in runs])
        ff = np.array([r[k]["ff"] for r in runs])
        sv = np.array([r[k].get("save_vs_slow", np.nan) for r in runs])
        sr = np.array([r[k]["slow_rate"] for r in runs])
        calls = [r[k].get("audit_calls") for r in runs if r[k].get("audit_calls") is not None]
        out[k] = {"acc_mean": float(acc.mean()), "acc_p05": pct(acc, 5), "acc_p95": pct(acc, 95),
                  "ff_mean": float(ff.mean()), "save_mean": float(np.nanmean(sv)) if np.isfinite(sv).any() else float("nan"),
                  "slow_rate_mean": float(sr.mean()),
                  "audit_calls_mean": float(np.mean(calls)) if calls else None}
        if "Random @ matched slow rate" in keys and k != "Random @ matched slow rate":
            rnd = np.array([r["Random @ matched slow rate"]["acc"] for r in runs])
            out[k]["gain_vs_random_mean"] = float((acc - rnd).mean())
            out[k]["p_beats_random"] = float((acc > rnd).mean())
    return out


# ------------------------------------------------------------------------- main
def main():
    rel, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    R = {}

    main_pair = load_main(rel)
    qwen = load_qwen(rel, "small", "large")
    qwen_mid = load_qwen(rel, "small", "mid")
    qwen_mid_large = load_qwen(rel, "mid", "large")
    lb = load_livebench(rel)
    tiers = qwen_tier_table(rel)

    # ---------- 1. shift quantification
    envs5 = ["A", "B", "C", "D", "E_test"]
    R["shift_main"] = shift_quantification(main_pair, envs5, FAM, rng)
    R["shift_qwen"] = shift_quantification(qwen, envs5, FAM, rng)
    R["shift_qwen_E_audit_cells"] = shift_quantification(qwen.envs("E_audit"), ["E_audit"], FAM, rng, B=1000, P=10)["cells"]
    R["shift_main_E_audit_cells"] = shift_quantification(main_pair.envs("E_audit"), ["E_audit"], FAM, rng, B=1000, P=10)["cells"]
    # LiveBench: categories observed in more than one release
    lb_cats = ["language", "math", "reasoning"]
    lb_envs = ["2024-06-24", "2024-07-26", "2024-08-31", "2024-11-25"]
    R["shift_livebench"] = shift_quantification(lb, lb_envs, lb_cats, rng)
    # LiveBench full category / subtask table
    rows = []
    for level, keyf in [("category", lambda i: lb.fam[i]), ("subtask", lambda i: lb.ids[i].split("|")[1])]:
        groups = collections.defaultdict(list)
        for i in range(len(lb.fam)):
            groups[(lb.fam[i], keyf(i))].append(i)
        for (cat, key), ii in sorted(groups.items()):
            ii = np.array(ii)
            bb = lb.b[ii]
            boots = bb[rng.integers(0, len(bb), size=(4000, len(bb)))].mean(axis=1)
            rows.append({"level": level, "category": cat, "group": key, "n": int(len(ii)),
                         "release": ";".join(sorted(set(lb.env[ii]))),
                         "score_35b": float(lb.yl[ii].mean()), "score_397b": float(lb.yu[ii].mean()),
                         "benefit": float(bb.mean()), "lo": pct(boots, 2.5), "hi": pct(boots, 97.5),
                         "wins_397b": int((bb > 0).sum()), "wins_35b": int((bb < 0).sum())})
    bb = lb.b
    boots = bb[rng.integers(0, len(bb), size=(4000, len(bb)))].mean(axis=1)
    rows.append({"level": "all", "category": "all", "group": "all", "n": int(len(bb)), "release": "all",
                 "score_35b": float(lb.yl.mean()), "score_397b": float(lb.yu.mean()), "benefit": float(bb.mean()),
                 "lo": pct(boots, 2.5), "hi": pct(boots, 97.5), "wins_397b": int((bb > 0).sum()), "wins_35b": int((bb < 0).sum())})
    R["livebench_table"] = rows

    # Qwen environment-level CIs (35B->397B, 35B->122B, 122B->397B)
    qrows = []
    for pr in [qwen, qwen_mid, qwen_mid_large]:
        for e in ["A", "B", "C", "D", "E_audit", "E_test", "ALL"]:
            s = pr if e == "ALL" else pr.envs(e)
            bb = s.b
            boots = bb[rng.integers(0, len(bb), size=(4000, len(bb)))].mean(axis=1)
            qrows.append({"pair": pr.name, "env": e, "n": int(len(bb)), "acc_lower": float(s.yl.mean()),
                          "acc_upper": float(s.yu.mean()), "benefit": float(bb.mean()), "lo": pct(boots, 2.5),
                          "hi": pct(boots, 97.5), "n_pos": int((bb > 0).sum()), "n_neg": int((bb < 0).sum())})
    R["qwen_env_table"] = qrows

    # ---------- 2. E locked: uncertainty of the headline result
    audit = main_pair.envs("E_audit")
    test = main_pair.envs("E_test")
    dk = np.maximum(fam_dk(audit), 0)
    fast = one_fast(fam_tau(audit), dk)
    base = evaluate_family(test, fast)
    is_fast = np.array([dict(zip(FAM, fast))[f] for f in test.fam])
    y = np.where(is_fast, test.yl, test.yu)
    k = np.where(is_fast, test.kl, test.ku)
    n = len(y)
    Bt = 10000
    bi = rng.integers(0, n, size=(Bt, n))
    acc_b = y[bi].mean(1)
    save_b = 1 - k[bi].sum(1) / test.ku[bi].sum(1)
    rnd_b = 0.8 * test.yu[bi].mean(1) + 0.2 * test.yl[bi].mean(1)
    rnd_tok_b = 0.8 * test.ku[bi].mean(1) + 0.2 * test.kl[bi].mean(1)
    gap_b = acc_b - rnd_b
    tokadv_b = 1 - k[bi].mean(1) / rnd_tok_b
    cp = clopper_pearson(int(y.sum()), n)
    R["E_locked"] = {
        "fast_family": [f for f, m in zip(FAM, fast) if m], "result": base,
        "acc_clopper_pearson": cp,
        "acc_boot": [pct(acc_b, 2.5), pct(acc_b, 97.5)],
        "save_vs_slow_boot": [pct(save_b, 2.5), pct(save_b, 97.5)],
        "gain_vs_random_boot": [pct(gap_b, 2.5), pct(gap_b, 97.5)], "gain_vs_random": float(y.mean() - (0.8 * test.yu.mean() + 0.2 * test.yl.mean())),
        "token_adv_vs_random_boot": [pct(tokadv_b, 2.5), pct(tokadv_b, 97.5)],
        "loss_vs_slow_exact_upper95": clopper_pearson(int(n - y.sum()), n, alpha=0.10)[1],
    }
    # audit-family statistics and certification arithmetic
    fstats = []
    for f in FAM:
        bb = audit.b[audit.fam == f]
        fstats.append({"family": f, "n": int(len(bb)), "tau": float(bb.mean()), "ucb_wald": ucb_net(bb),
                       "ucb_hoeffding": hoeffding_ucb(bb), "p_tau_le_gamma": posterior_prob_le(bb, GAMMA, 40000, rng),
                       "dk": float(np.mean((audit.ku - audit.kl)[audit.fam == f]))})
    R["E_audit_family_stats"] = fstats
    # n needed for Hoeffding certification of a family with true tau t at gamma: n >= 2 log(1/eta) / (gamma - t)^2
    R["hoeffding_n_required"] = {f"tau={t:.2f}": math.ceil(2 * math.log(20) / (GAMMA - t) ** 2) for t in [0.0, 0.03, 0.05, 0.10]}

    # audit bootstrap (resample audit within family), fixed test
    picks = collections.Counter()
    accs, saves, ffs = [], [], []
    acc2 = []
    a_by = {f: np.where(audit.fam == f)[0] for f in FAM}
    for rep in range(5000):
        idx = np.concatenate([a_by[f][rng.integers(0, len(a_by[f]), size=len(a_by[f]))] for f in FAM])
        a = Pair(audit.name, audit.env[idx], audit.fam[idx], audit.X[idx], audit.yl[idx], audit.yu[idx], audit.kl[idx], audit.ku[idx], audit.ids[idx])
        fm = one_fast(fam_tau(a), np.maximum(fam_dk(a), 0))
        picks["+".join(f for f, m in zip(FAM, fm) if m) or "none"] += 1
        r = evaluate_family(test, fm)
        accs.append(r["acc"]); saves.append(r["save_vs_slow"]); ffs.append(r["ff"])
        # two-level: also resample test
        ti = rng.integers(0, len(test.fam), size=len(test.fam))
        tt = Pair(test.name, test.env[ti], test.fam[ti], test.X[ti], test.yl[ti], test.yu[ti], test.kl[ti], test.ku[ti], test.ids[ti])
        acc2.append(evaluate_family(tt, fm)["acc"])
    R["E_audit_bootstrap"] = {"reps": 5000, "pick_freq": {k2: v / 5000 for k2, v in picks.most_common()},
                              "acc_mean": float(np.mean(accs)), "acc_p05": pct(accs, 5), "acc_p50": pct(accs, 50), "acc_p95": pct(accs, 95),
                              "save_mean": float(np.mean(saves)), "save_p05": pct(saves, 5), "save_p95": pct(saves, 95),
                              "ff_mean": float(np.mean(ffs)), "ff_p95": pct(ffs, 95),
                              "two_level_acc_ci": [pct(acc2, 2.5), pct(acc2, 97.5)],
                              "p_acc_ge_random": float(np.mean(np.array(accs) > 0.8 * test.yu.mean() + 0.2 * test.yl.mean()))}
    # gamma / alpha sensitivity
    sens = []
    for g in [0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25]:
        for kf, sr in [(1, 0.8), (2, 0.6)]:
            fm = one_fast(fam_tau(audit), dk, g, kf)
            r = evaluate_family(test, fm)
            sens.append({"gamma": g, "max_fast_families": kf, "alpha": sr, "fast": "+".join(f for f, m in zip(FAM, fm) if m) or "none", **r})
        fm = threshold_fast(fam_tau(audit), g)
        r = evaluate_family(test, fm)
        sens.append({"gamma": g, "max_fast_families": "unconstrained", "alpha": None, "fast": "+".join(f for f, m in zip(FAM, fm) if m) or "none", **r})
    R["E_gamma_alpha_sensitivity"] = sens

    # ---------- 3. E locked baselines on the same 100 target samples
    hist_E = main_pair.envs("A", "B", "C")
    R["E_locked_baselines"] = family_policy_suite(hist_E, audit, test, rng, hist_envs=["A", "B", "C"])
    # audit overhead
    fast_tok_audit = float(audit.kl.sum())
    slow_tok_audit = float(audit.ku.sum())
    save_per_query = float(test.ku.mean() - base["tokens"])
    R["E_audit_overhead"] = {"audit_items": int(len(audit.fam)), "paired_calls": int(2 * len(audit.fam)),
                             "audit_fast_tokens": fast_tok_audit, "audit_slow_tokens": slow_tok_audit,
                             "audit_total_tokens": fast_tok_audit + slow_tok_audit,
                             "saving_per_routed_query_vs_slow": save_per_query,
                             "breakeven_queries_gross": (fast_tok_audit + slow_tok_audit) / save_per_query,
                             "breakeven_queries_net_of_served_slow": fast_tok_audit / save_per_query,
                             "net_saving_on_500_test_gross": float((test.ku.sum() - k.sum() - fast_tok_audit - slow_tok_audit) / test.ku.sum()),
                             "net_saving_on_600_incl_audit_served_slow": float((test.ku.sum() + slow_tok_audit - k.sum() - slow_tok_audit - fast_tok_audit) / (test.ku.sum() + slow_tok_audit))}

    # ---------- 4. audit size / composition replay on pooled E (600 items)
    Epool = main_pair.envs("E_audit", "E_test")
    comp = []
    for m in [50, 100, 150, 200]:
        for scheme in ["balanced", "simple_random", "skewed_dirichlet"]:
            runs = []
            pickopt = 0
            for rep in range(1000):
                if scheme == "balanced":
                    a, t = split_balanced(Epool, m // 5, rng)
                else:
                    if scheme == "simple_random":
                        sel = rng.permutation(len(Epool.fam))[:m]
                    else:
                        w = rng.dirichlet([1.0] * 5)
                        cnt = rng.multinomial(m, w)
                        cnt = np.minimum(cnt, 119)
                        sel = np.concatenate([np.where(Epool.fam == f)[0][rng.permutation(120)[:c]] for f, c in zip(FAM, cnt)])
                    msk = np.zeros(len(Epool.fam), dtype=bool)
                    msk[sel] = True
                    a, t = Epool.sub(msk), Epool.sub(~msk)
                tau = np.array([a.b[a.fam == f].mean() if (a.fam == f).sum() > 0 else np.nan for f in FAM])
                d = np.array([np.mean((a.ku - a.kl)[a.fam == f]) if (a.fam == f).sum() > 0 else 0 for f in FAM])
                fm = one_fast(tau, np.maximum(d, 0))
                pickopt += int(fm[FAM.index("optimization")])
                r = evaluate_family(t, fm)
                rm = random_matched(t, 0.8)
                runs.append({"p": r, "Random @ matched slow rate": {**rm, "save_vs_slow": 1 - rm["tokens"] / t.ku.mean()}})
            agg = aggregate_runs(runs)["p"]
            comp.append({"audit_n": m, "scheme": scheme, **agg, "p_pick_optimization": pickopt / 1000})
    R["E_pooled_audit_composition"] = comp

    # near-boundary definition sensitivity on locked E (hist A-C), plus radius-widened variant
    nb = []
    hist_tau = fam_tau(hist_E)
    n_hist = np.array([(hist_E.fam == f).sum() for f in FAM])
    radius = 1.0 / np.sqrt(n_hist)  # sd of a bounded net benefit is at most 1
    inc = one_fast(hist_tau, dk)
    audit_tau = fam_tau(audit)
    for d0 in [0.0, 0.025, 0.05, 0.075, 0.10, 0.15]:
        for label, extra in [("plug-in margin", 0 * radius), ("margin + 1.96 x historical s.e.", 1.96 * radius)]:
            for incl_inc in [True, False]:
                audited = (np.abs(hist_tau - GAMMA) <= d0 + extra + 1e-12) | (inc if incl_inc else False)
                mixed = np.where(audited, audit_tau, hist_tau)
                r = evaluate_family(test, one_fast(mixed, dk))
                nb.append({"delta0": d0, "width": label, "incumbent": incl_inc, "calls": int(20 * audited.sum()),
                           "audited": "+".join(f for f, mm in zip(FAM, audited) if mm), **r})
    R["E_near_boundary_sensitivity"] = nb

    # ---------- 5. independent target replications (1,000 balanced 100/rest splits)
    rep_specs = [
        ("Primary pair, target B (history A)", main_pair, ["A"], ["B"]),
        ("Primary pair, target D (history A-C)", main_pair, ["A", "B", "C"], ["D"]),
        ("Primary pair, target E (history A-D)", main_pair, ["A", "B", "C", "D"], ["E_audit", "E_test"]),
        ("Qwen pair, target B (history A)", qwen, ["A"], ["B"]),
        ("Qwen pair, target C (history A-B)", qwen, ["A", "B"], ["C"]),
        ("Qwen pair, target D (history A-C)", qwen, ["A", "B", "C"], ["D"]),
        ("Qwen pair, target E (history A-D)", qwen, ["A", "B", "C", "D"], ["E_audit", "E_test"]),
    ]
    reps_out = {}
    for name, pr, hist_e, tgt_e in rep_specs:
        hist = pr.envs(*hist_e)
        tgt = pr.envs(*tgt_e)
        runs = []
        for rep in range(1000 if "Primary" in name else 500):
            a, t = split_balanced(tgt, 20, rng)
            runs.append(family_policy_suite(hist, a, t, rng, hist_envs=hist_e, with_items=("Primary" in name)))
        reps_out[name] = aggregate_runs(runs)
    R["replications"] = reps_out

    # ---------- 6. margin calibration (Prop. 2): estimable shift budget
    mc = []
    mcal_budget = {}
    for pr_name, pr in [("primary", main_pair), ("qwen", qwen)]:
        hist = pr.envs("A", "B", "C")
        tau_e = {e: fam_tau(pr.envs(e)) for e in ["A", "B", "C"]}
        tau_h = fam_tau(hist)
        loeo = np.array([[abs(tau_e[e][i] - np.mean([tau_e[x][i] for x in ["A", "B", "C"] if x != e])) for e in ["A", "B", "C"]] for i in range(5)])
        d_fam = loeo.max(axis=1)                      # per-family LOEO budget
        d_pool = float(np.quantile(loeo.ravel(), 0.9))  # pooled 90% LOEO budget
        for tgt in ["D", "E_test"]:
            tau_t = fam_tau(pr.envs(tgt))
            for i, f in enumerate(FAM):
                vH, vT = tau_h[i] - GAMMA, tau_t[i] - GAMMA
                flip = (vH > 0) != (vT > 0)
                mc.append({"pair": pr_name, "target": tgt, "family": f, "vH": float(vH), "vT": float(vT),
                           "shift": float(abs(vT - vH)), "flip": bool(flip), "regret": float(abs(vT)) if flip else 0.0,
                           "delta_loeo_family": float(d_fam[i]), "delta_loeo_pooled90": d_pool})
        mcal_budget[pr_name] = {"pooled90": d_pool, "by_family": dict(zip(FAM, map(float, d_fam)))}
    mcal = {"cells": mc, "budgets": mcal_budget}
    for col in ["delta_loeo_family", "delta_loeo_pooled90"]:
        cov = np.mean([c["shift"] <= c[col] + 1e-12 for c in mc])
        flips_in = [c for c in mc if c["flip"]]
        captured = np.mean([abs(c["vH"]) <= c[col] + 1e-12 for c in flips_in]) if flips_in else float("nan")
        bound = np.mean([c[col] * (abs(c["vH"]) <= c[col]) for c in mc])
        realized = np.mean([c["regret"] for c in mc])
        audit_frac = np.mean([abs(c["vH"]) <= c[col] for c in mc])
        mcal[col] = {"coverage_of_realized_shift": float(cov), "flips_inside_margin": float(captured),
                     "n_flips": len(flips_in), "bound_mean": float(bound), "realized_regret_mean": float(realized),
                     "fraction_cells_inside_margin": float(audit_frac)}
    # oracle-delta curve: for delta0 grid
    curve = []
    for d0 in [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]:
        inside = [abs(c["vH"]) <= d0 + 1e-12 for c in mc]
        flips = [c["flip"] for c in mc]
        reg = np.array([c["regret"] for c in mc])
        curve.append({"delta0": d0, "cells_inside": float(np.mean(inside)),
                      "flips_captured": float(sum(i and f for i, f in zip(inside, flips)) / max(1, sum(flips))),
                      "regret_captured": float(reg[np.array(inside)].sum() / max(1e-12, reg.sum())),
                      "cells_with_shift_le_delta0": float(np.mean([c["shift"] <= d0 + 1e-12 for c in mc]))})
    mcal["delta_curve"] = curve
    mcal["note"] = "delta_loeo_pooled90 is the 90th percentile of leave-one-environment-out family shifts over A-C, pooled across families"
    R["margin_calibration"] = mcal

    # ---------- 7. serving-change axis: endpoint swaps after calibration (E test)
    def cross_pair(lower_src, upper_tier, env_names):
        rows = []
        main_by = {}
        for i in range(len(main_pair.ids)):
            if main_pair.env[i] in env_names:
                main_by[main_pair.ids[i]] = i
        for iid, u in tiers.items():
            if u.get(upper_tier) is None:
                continue
            env = u[upper_tier]["source_env"]
            if env not in env_names:
                continue
            if lower_src == "deepseek":
                if iid not in main_by:
                    continue
                i = main_by[iid]
                yl, kl = main_pair.yl[i], main_pair.kl[i]
            else:
                if lower_src not in u:
                    continue
                yl, kl = int(u[lower_src]["correct"]), tok(u[lower_src])
            r = u[upper_tier]
            rows.append((env, r["family"], feat(r), yl, int(r["correct"]), kl, tok(r), iid))
        env, fam, X, yl, yu, kl, ku, ids = zip(*rows)
        return Pair(f"{lower_src}->{upper_tier}", env, fam, X, yl, yu, kl, ku, ids)

    swaps = []
    deployed = {
        "Primary-pair E-audit route (optimization fast)": one_fast(fam_tau(audit), dk),
        "Qwen-pair E-audit route": threshold_fast(fam_tau(qwen.envs("E_audit"))),
    }
    serving = {
        "Prompt/budget drift: 397B with verification prompt, 8,192 cap": cross_pair("deepseek", "large", ["E_test"]),
        "Drift + escalation endpoint swapped to 122B": cross_pair("deepseek", "mid", ["E_test"]),
        "Drift + baseline endpoint swapped to 35B": cross_pair("small", "large", ["E_test"]),
        "Drift + both endpoints swapped (35B -> 122B)": cross_pair("small", "mid", ["E_test"]),
    }
    serving_audit = {
        "Prompt/budget drift: 397B with verification prompt, 8,192 cap": cross_pair("deepseek", "large", ["E_audit"]),
        "Drift + escalation endpoint swapped to 122B": cross_pair("deepseek", "mid", ["E_audit"]),
        "Drift + baseline endpoint swapped to 35B": cross_pair("small", "large", ["E_audit"]),
        "Drift + both endpoints swapped (35B -> 122B)": cross_pair("small", "mid", ["E_audit"]),
    }
    route0 = deployed["Primary-pair E-audit route (optimization fast)"]
    for sname, sp in serving.items():
        taus = fam_tau(sp)
        r_frozen = evaluate_family(sp, route0)
        sa = serving_audit[sname]
        re_route = one_fast(fam_tau(sa), np.maximum(fam_dk(sa), 0))
        r_re = evaluate_family(sp, re_route)
        best_acc = max(evaluate_family(sp, np.eye(5, dtype=bool)[i])["acc"] for i in range(5))
        swaps.append({"serving": sname, "n": int(len(sp.fam)), "tau_by_family": dict(zip(FAM, map(float, taus))),
                      "frozen_route": "optimization", "frozen_acc": r_frozen["acc"], "frozen_ff": r_frozen["ff"],
                      "reaudit_route": "+".join(f for f, m in zip(FAM, re_route) if m) or "none",
                      "reaudit_acc": r_re["acc"], "reaudit_ff": r_re["ff"],
                      "uniform_upper_acc": float(sp.yu.mean()), "uniform_lower_acc": float(sp.yl.mean()),
                      "best_single_fast_family_acc": best_acc,
                      "random80_acc": random_matched(sp, 0.8)["acc"]})
    R["serving_swaps"] = swaps

    # ---------- 8. LiveBench routing: release-shift test and 100-item audit protocol
    lbres = {}
    hist = lb.sub(lb.env != "2024-11-25")
    tgt = lb.sub(lb.env == "2024-11-25")
    tcats = sorted(set(tgt.fam))
    for g in [0.025, 0.05, 0.075, 0.10]:
        htau = {c: hist.b[hist.fam == c].mean() for c in set(hist.fam)}
        hist_fast = {c: (c in htau and htau[c] <= g + 1e-12) for c in tcats}  # unseen category -> slow
        runs = []
        for rep in range(1000):
            a_idx, t_idx = [], []
            for c in tcats:
                ii = np.where(tgt.fam == c)[0]
                ii = ii[rng.permutation(len(ii))]
                a_idx.extend(ii[:20]); t_idx.extend(ii[20:])
            msk = np.zeros(len(tgt.fam), dtype=bool); msk[a_idx] = True
            a, t = tgt.sub(msk), tgt.sub(~msk)
            atau = {c: a.b[a.fam == c].mean() for c in tcats}
            res_h = evaluate_items(t, np.array([hist_fast[c] for c in t.fam]))
            fa = {c: atau[c] <= g + 1e-12 for c in tcats}
            res_a = evaluate_items(t, np.array([fa[c] for c in t.fam]))
            # focused: audit only historical-fast or near-boundary or unseen categories
            foc = {c: (c not in htau) or hist_fast[c] or abs(htau[c] - g) <= 0.025 for c in tcats}
            ff_ = {c: (atau[c] if foc[c] else htau[c]) <= g + 1e-12 for c in tcats}
            res_f = evaluate_items(t, np.array([ff_[c] for c in t.fam]))
            res_f["audit_calls"] = 20 * sum(foc.values())
            out_r = {"Historical releases only": res_h, "Target audit (20 per category)": {**res_a, "audit_calls": 60},
                     "Focused target audit": res_f}
            for kk in list(out_r):
                rr = out_r[kk]
                rr["rand_acc"] = random_matched(t, rr["slow_rate"])["acc"]
            out_r["Uniform 397B"] = evaluate_items(t, np.zeros(len(t.fam), dtype=bool)); out_r["Uniform 397B"]["rand_acc"] = out_r["Uniform 397B"]["acc"]
            runs.append(out_r)
        agg = {}
        for kk in runs[0]:
            acc = np.array([r[kk]["acc"] for r in runs]); ra = np.array([r[kk]["rand_acc"] for r in runs])
            agg[kk] = {"score_mean": float(acc.mean()), "score_p05": pct(acc, 5), "score_p95": pct(acc, 95),
                       "small_share": float(np.mean([1 - r[kk]["slow_rate"] for r in runs])),
                       "loss_mean": float(np.mean([r[kk]["ff_loss"] for r in runs])),
                       "gain_vs_random": float((acc - ra).mean()),
                       "audit_calls": float(np.mean([r[kk].get("audit_calls", 0) for r in runs]))}
        lbres[f"gamma={g}"] = {"historical_category_benefit": {c: float(v) for c, v in htau.items()},
                               "target_category_benefit": {c: float(tgt.b[tgt.fam == c].mean()) for c in tcats},
                               "historical_fast": [c for c in tcats if hist_fast[c]], "policies": agg}
    # full-suite 100-item audit protocol (20 per category over all 772 items)
    allcats = sorted(set(lb.fam))
    full = {}
    for g in [0.025, 0.05, 0.10]:
        runs = []
        for rep in range(1000):
            a_idx = []
            for c in allcats:
                ii = np.where(lb.fam == c)[0]
                a_idx.extend(ii[rng.permutation(len(ii))[:20]])
            msk = np.zeros(len(lb.fam), dtype=bool); msk[a_idx] = True
            a, t = lb.sub(msk), lb.sub(~msk)
            fa = {c: a.b[a.fam == c].mean() <= g + 1e-12 for c in allcats}
            r = evaluate_items(t, np.array([fa[c] for c in t.fam]))
            r["rand_acc"] = random_matched(t, r["slow_rate"])["acc"]
            # hindsight oracle category policy at this gamma
            orc = {c: t.b[t.fam == c].mean() <= g + 1e-12 for c in allcats}
            r["oracle_acc"] = evaluate_items(t, np.array([orc[c] for c in t.fam]))["acc"]
            r["oracle_small"] = 1 - evaluate_items(t, np.array([orc[c] for c in t.fam]))["slow_rate"]
            r["agree"] = float(np.mean([fa[c] == orc[c] for c in allcats]))
            r["uniform_large"] = float(t.yu.mean())
            runs.append(r)
        full[f"gamma={g}"] = {k2: float(np.mean([r[k2] for r in runs])) for k2 in ["acc", "slow_rate", "rand_acc", "oracle_acc", "oracle_small", "agree", "uniform_large", "ff_loss"]}
        full[f"gamma={g}"]["acc_p05"] = pct([r["acc"] for r in runs], 5)
        full[f"gamma={g}"]["p_beats_random"] = float(np.mean([r["acc"] > r["rand_acc"] for r in runs]))
    lbres["full_suite_audit"] = full
    R["livebench_routing"] = lbres

    (out / "revision_results.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    print("wrote", out / "revision_results.json")


if __name__ == "__main__":
    main()
