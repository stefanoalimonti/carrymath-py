"""Exact enumeration over 2^{2K} bit configurations.

The workhorse for computing exact expectations and covariances
in the carry chain model. For K free bit-pairs (after fixing g_0=h_0=1),
we enumerate all 2^{2K} configurations and accumulate statistics.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def _decode_bits(cfg: int, K: int) -> Tuple[List[int], List[int]]:
    """Decode a configuration index into (g, h) bit sequences.

    g[0] = h[0] = 1 (fixed MSBs).
    g[1..K] and h[1..K] are extracted from cfg.
    """
    g = [1] + [((cfg >> (2 * K - 1 - i)) & 1) for i in range(K)]
    h = [1] + [((cfg >> (K - 1 - i)) & 1) for i in range(K)]
    return g, h


def enumerate_exact(
    K: int,
    observables: Optional[Dict[str, Callable]] = None,
) -> Dict[str, Fraction]:
    """Enumerate all 2^{2K} configurations and compute exact expectations.

    Parameters
    ----------
    K : number of free bit pairs (total positions = K+1)
    observables : dict of {name: func(g, h, carries, convs) -> Fraction}
        Functions that compute an observable from a single configuration.
        Default observables: E_carry (per-position), E_conv, c1 (trace anomaly).

    Returns
    -------
    results : dict of exact Fraction expectations.

    Example
    -------
    >>> r = enumerate_exact(3)
    >>> float(r['c1'])  # trace anomaly at K=3
    0.125
    """
    n_configs = 1 << (2 * K)
    D = K + 1

    if observables is None:
        E_carry = [Fraction(0)] * (D + 1)
        E_conv = [Fraction(0)] * D
        for cfg in range(n_configs):
            g, h = _decode_bits(cfg, K)
            carries = [0]
            convs = []
            for j in range(D):
                cv = sum(g[i] * h[j - i] for i in range(j + 1) if i < D and j - i < D)
                convs.append(cv)
                carries.append((cv + carries[-1]) // 2)
            for j in range(D + 1):
                E_carry[j] += Fraction(carries[j])
            for j in range(D):
                E_conv[j] += Fraction(convs[j])
        N = Fraction(n_configs)
        E_carry = [e / N for e in E_carry]
        E_conv = [e / N for e in E_conv]
        c1 = E_carry[D - 1]
        return {
            "E_carry": E_carry,
            "E_conv": E_conv,
            "c1": c1,
            "K": K,
            "D": D,
        }

    accum: dict[str, Fraction] = {name: Fraction(0) for name in observables}
    N = Fraction(n_configs)
    for cfg in range(n_configs):
        g, h = _decode_bits(cfg, K)
        carries = [0]
        convs = []
        for j in range(D):
            cv = sum(g[i] * h[j - i] for i in range(j + 1) if i < D and j - i < D)
            convs.append(cv)
            carries.append((cv + carries[-1]) // 2)
        for name, func in observables.items():
            accum[name] += func(g, h, carries, convs)
    return {name: val / N for name, val in accum.items()}


def enumerate_numpy(K: int) -> Dict[str, np.ndarray]:
    """Fast numpy-based enumeration. Returns float arrays (not exact).

    Uses floating-point accumulators instead of exact Fraction arithmetic.

    Returns
    -------
    dict with keys:
        'carries': (D+1,) array of E[carry_j]
        'convs': (D,) array of E[conv_j]
        'c1': float trace anomaly
        'cov_carry_conv': (D,) array of Cov(carry_j, conv_j)
    """
    n_configs = 1 << (2 * K)
    D = K + 1

    E_carry = np.zeros(D + 1)
    E_conv = np.zeros(D)
    E_cc = np.zeros(D)
    E_c2 = np.zeros(D + 1)
    E_v2 = np.zeros(D)

    for cfg in range(n_configs):
        g = [1] + [((cfg >> (2 * K - 1 - i)) & 1) for i in range(K)]
        h = [1] + [((cfg >> (K - 1 - i)) & 1) for i in range(K)]
        carries = [0]
        for j in range(D):
            cv = sum(g[i] * h[j - i] for i in range(j + 1) if i < D and j - i < D)
            carries.append((cv + carries[-1]) // 2)
            E_conv[j] += cv
            E_cc[j] += carries[j] * cv
            E_v2[j] += cv * cv
        for j in range(D + 1):
            E_carry[j] += carries[j]
            E_c2[j] += carries[j] ** 2

    N = float(n_configs)
    E_carry /= N
    E_conv /= N
    E_cc /= N
    E_c2 /= N
    E_v2 /= N

    cov = E_cc - E_carry[:D] * E_conv

    return {
        "carries": E_carry,
        "convs": E_conv,
        "c1": float(E_carry[D - 1]),
        "cov_carry_conv": cov,
        "var_carry": E_c2 - E_carry**2,
        "var_conv": E_v2 - E_conv**2,
        "K": K,
        "D": D,
    }
