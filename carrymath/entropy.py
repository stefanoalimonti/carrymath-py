"""BFS entropy computation for carry-chain factorization.

Computes S₀(k) = log₂(number of valid (p,q,carry) states at position k)
for a given semiprime N. The entropy curve reveals the phantom state
decomposition S₀ = τ(M_k) + Φ(k).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional


def bfs_states(N: int, max_pos: Optional[int] = None) -> List[int]:
    """BFS enumeration of valid carry-chain states at each position.

    At position k, a valid state is a triple (p mod 2^k, q mod 2^k, carry)
    consistent with N mod 2^k = (p*q) mod 2^k.

    Parameters
    ----------
    N : the semiprime to factorize
    max_pos : maximum position (default: bit length of N)

    Returns
    -------
    counts : list where counts[k] = number of valid states at position k
    """
    if max_pos is None:
        max_pos = N.bit_length()

    states: set[tuple[int, int, int]] = set()
    for a in range(2):
        for b in range(2):
            prod_bit = a * b
            if prod_bit % 2 == N % 2:
                carry_out = prod_bit // 2
                states.add((a, b, carry_out))

    counts = [len(states)]

    for k in range(1, max_pos):
        next_states: set[tuple[int, int, int]] = set()
        for pp, qq, cin in states:
            for a in range(2):
                for b in range(2):
                    new_p = pp | (a << k)
                    new_q = qq | (b << k)
                    col_sum = 0
                    for i in range(k + 1):
                        pi = (new_p >> i) & 1
                        qi = (new_q >> (k - i)) & 1
                        col_sum += pi * qi
                    col_sum += cin
                    digit = col_sum % 2
                    expected_digit = (N >> k) & 1
                    if digit == expected_digit:
                        carry_out = col_sum // 2
                        next_states.add((new_p, new_q, carry_out))
        states = next_states
        counts.append(len(states))

    return counts


def bfs_simple(N: int, max_pos: Optional[int] = None) -> List[int]:
    """Simplified BFS: enumerate valid (p_partial, q_partial) pairs at each position.

    Uses the modular constraint p*q ≡ N mod 2^{k+1} directly.
    Simpler but slower for large N.

    Parameters
    ----------
    N : the semiprime
    max_pos : maximum position (default: bit length of N)

    Returns
    -------
    counts : valid pair counts at each position
    """
    if max_pos is None:
        max_pos = N.bit_length()

    counts = []
    for k in range(1, max_pos + 1):
        mod = 1 << k
        n_mod = N % mod
        count = 0
        for p in range(1, mod, 2):
            q = (n_mod * pow(p, -1, mod)) % mod if math.gcd(p, mod) == 1 else -1
            if q >= 0 and (p * q) % mod == n_mod and q % 2 == 1:
                count += 1
        counts.append(count)
    return counts


def entropy_curve(N: int, max_pos: Optional[int] = None) -> List[float]:
    """Compute H(k) = log₂(S₀(k)) at each position.

    Returns
    -------
    H : list of floats, H[k] = log2(count[k]) or 0 if count is 0.
    """
    counts = bfs_simple(N, max_pos)
    return [math.log2(c) if c > 0 else 0.0 for c in counts]


def divisor_component(N: int, max_pos: Optional[int] = None) -> List[int]:
    """Compute τ(M_k) = #{d | M_k : 1 ≤ d < 2^k, 1 ≤ M_k/d < 2^k}
    where M_k = N mod 2^{k+1}.

    This is the "genuine factorization" component of S₀.
    """
    if max_pos is None:
        max_pos = N.bit_length()

    tau = []
    for k in range(1, max_pos + 1):
        mod = 1 << k
        m_k = N % (mod * 2)
        count = 0
        for d in range(1, mod):
            if m_k % d == 0:
                q = m_k // d
                if 1 <= q < mod:
                    count += 1
        tau.append(count)
    return tau


def phantom_counts(N: int, max_pos: Optional[int] = None) -> List[int]:
    """Compute Φ(k) = S₀(k) - τ(M_k), the phantom state count.

    Phantoms are locally valid carry-chain states that correspond to
    no actual factorization.
    """
    s0 = bfs_simple(N, max_pos)
    tau = divisor_component(N, max_pos)
    return [s - t for s, t in zip(s0, tau)]


def entropy_profile(N: int, max_pos: Optional[int] = None) -> Dict[str, List]:
    """Complete entropy profile: S₀, τ, Φ, and H.

    Returns
    -------
    dict with keys: 'S0', 'tau', 'phantom', 'H', 'phantom_ratio'
    """
    s0 = bfs_simple(N, max_pos)
    tau = divisor_component(N, max_pos)
    phi = [s - t for s, t in zip(s0, tau)]
    H = [math.log2(s) if s > 0 else 0.0 for s in s0]
    ratio = [p / s if s > 0 else 0.0 for p, s in zip(phi, s0)]
    return {
        "S0": s0,
        "tau": tau,
        "phantom": phi,
        "H": H,
        "phantom_ratio": ratio,
    }
