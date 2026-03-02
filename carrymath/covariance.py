"""Exact covariance computation for the carry chain.

Implements verification of:
- Lemma D: Cov(carry_j, g_i h_{j-i}) = 1/8 for all off-diagonal i ≠ j-i
- Main theorem: Cov(carry_j, conv_j) = (j-1)/8 for odd j >= 3
- Even-j corrections ε_j
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Tuple


def _enumerate_covariance_exact(
    K: int,
) -> Tuple[List[Fraction], List[Fraction], List[Fraction]]:
    """Enumerate all 2^{2K} configs and return exact carry/conv expectations.

    Returns (E_carry, E_conv, E_carry_conv) each indexed by position j.
    """
    n_configs = 1 << (2 * K)
    D = K + 1
    N = Fraction(n_configs)

    E_c = [Fraction(0)] * (D + 1)
    E_v = [Fraction(0)] * D
    E_cv = [Fraction(0)] * D

    for cfg in range(n_configs):
        g = [1] + [((cfg >> (2 * K - 1 - i)) & 1) for i in range(K)]
        h = [1] + [((cfg >> (K - 1 - i)) & 1) for i in range(K)]
        carries = [0]
        for j in range(D):
            cv = sum(g[i] * h[j - i] for i in range(j + 1) if i < D and j - i < D)
            carries.append((cv + carries[-1]) // 2)
            E_v[j] += Fraction(cv)
            E_cv[j] += Fraction(carries[j] * cv)
        for j in range(D + 1):
            E_c[j] += Fraction(carries[j])

    return [e / N for e in E_c], [e / N for e in E_v], [e / N for e in E_cv]


def bulk_covariance(K: int) -> Dict[str, object]:
    """Compute exact Cov(carry_j, conv_j) for all j at window size K.

    Returns
    -------
    dict with:
        'cov': list of Fraction — Cov(carry_j, conv_j) for j=0..D-1
        'E_carry': list of Fraction
        'E_conv': list of Fraction
        'K': int
        'D': int
        'formula_check': dict — {j: (j-1)/8} for odd j, showing match

    Example
    -------
    >>> r = bulk_covariance(4)
    >>> r['cov'][3] == Fraction(1, 4)  # Cov(c_3, v_3) = 2/8 = 1/4
    True
    """
    D = K + 1
    E_c, E_v, E_cv = _enumerate_covariance_exact(K)
    cov = [E_cv[j] - E_c[j] * E_v[j] for j in range(D)]

    formula_check = {}
    for j in range(3, D, 2):
        expected = Fraction(j - 1, 8)
        formula_check[j] = {
            "computed": cov[j],
            "formula": expected,
            "match": cov[j] == expected,
        }

    return {
        "cov": cov,
        "E_carry": E_c,
        "E_conv": E_v,
        "K": K,
        "D": D,
        "formula_check": formula_check,
    }


def offdiagonal_covariance(K: int) -> Dict[int, Dict[int, Fraction]]:
    """Compute Cov(carry_j, g_i h_{j-i}) for all (j, i) pairs.

    Verifies Lemma D: each off-diagonal pair should equal 1/8.

    Returns
    -------
    result[j][i] = Cov(carry_j, g_i h_{j-i}) as exact Fraction.

    Example
    -------
    >>> r = offdiagonal_covariance(4)
    >>> r[3][1] == Fraction(1, 8)  # off-diagonal: i=1, j-i=2, i != j-i
    True
    """
    n_configs = 1 << (2 * K)
    D = K + 1
    N = Fraction(n_configs)

    E_c = [Fraction(0)] * (D + 1)
    E_gh: dict[tuple[int, int], Fraction] = {}
    E_c_gh: dict[tuple[int, int], Fraction] = {}

    for cfg in range(n_configs):
        g = [1] + [((cfg >> (2 * K - 1 - i)) & 1) for i in range(K)]
        h = [1] + [((cfg >> (K - 1 - i)) & 1) for i in range(K)]
        carries = [0]
        for j in range(D):
            cv = sum(g[i] * h[j - i] for i in range(j + 1) if i < D and j - i < D)
            carries.append((cv + carries[-1]) // 2)
        for j in range(D + 1):
            E_c[j] += Fraction(carries[j])
        for j in range(1, D):
            for i in range(j + 1):
                if i < D and (j - i) < D:
                    key = (j, i)
                    prod = g[i] * h[j - i]
                    E_gh[key] = E_gh.get(key, Fraction(0)) + Fraction(prod)
                    E_c_gh[key] = E_c_gh.get(key, Fraction(0)) + Fraction(carries[j] * prod)

    E_c = [e / N for e in E_c]
    for key in E_gh:
        E_gh[key] /= N
        E_c_gh[key] /= N

    result: dict[int, dict[int, Fraction]] = {}
    for j in range(1, D):
        result[j] = {}
        for i in range(j + 1):
            if i < D and (j - i) < D:
                key = (j, i)
                cov = E_c_gh[key] - E_c[j] * E_gh[key]
                result[j][i] = cov

    return result


def verify_lemma_d(K: int) -> Dict[str, object]:
    """Verify Lemma D: Cov(carry_j, g_i h_{j-i}) = 1/8 for all off-diagonal pairs.

    Returns summary with per-pair results and overall verdict.
    """
    eighth = Fraction(1, 8)
    result = offdiagonal_covariance(K)
    D = K + 1

    all_offdiag_match = True
    details = []
    for j in sorted(result.keys()):
        for i in sorted(result[j].keys()):
            is_diagonal = i == j - i
            is_boundary = i == 0 or i == j
            cov = result[j][i]
            is_match = cov == eighth
            if not is_diagonal and not is_boundary and not is_match:
                all_offdiag_match = False
            details.append(
                {
                    "j": j,
                    "i": i,
                    "j-i": j - i,
                    "diagonal": is_diagonal,
                    "boundary": is_boundary,
                    "cov": cov,
                    "equals_1_8": is_match,
                }
            )

    return {
        "K": K,
        "D": D,
        "lemma_d_holds": all_offdiag_match,
        "details": details,
        "n_interior_offdiag": sum(1 for d in details if not d["diagonal"] and not d["boundary"]),
        "n_interior_offdiag_match": sum(
            1 for d in details if not d["diagonal"] and not d["boundary"] and d["equals_1_8"]
        ),
    }


def even_j_corrections(K: int) -> Dict[int, Fraction]:
    """Compute ε_j = Cov(carry_j, conv_j) - (j-1)/8 for even j.

    These corrections arise from the diagonal term g_{j/2} h_{j/2}
    breaking the Parity Lemma.

    Returns dict {j: ε_j} for even j in range.
    """
    r = bulk_covariance(K)
    corrections: dict[int, Fraction] = {}
    D = r["D"]
    for j in range(2, D, 2):
        corrections[j] = r["cov"][j] - Fraction(j - 1, 8)
    return corrections


def induction_step(K: int) -> Dict[int, Fraction]:
    """Compute ΔCov = Cov(c_{j+2}, v_{j+2}) - Cov(c_j, v_j) for all j.

    For odd j -> odd j+2, this should be exactly 1/4.

    Returns dict {j: ΔCov}.
    """
    r = bulk_covariance(K)
    D = r["D"]
    deltas: dict[int, Fraction] = {}
    for j in range(1, D - 2):
        deltas[j] = r["cov"][j + 2] - r["cov"][j]
    return deltas
