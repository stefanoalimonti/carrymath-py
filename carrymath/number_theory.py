"""Number-theoretic utilities: totient, primitive roots, Legendre symbol, characters."""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

from carrymath.digits import eval_poly_mod


def euler_totient(n: int) -> int:
    """Euler's totient function phi(n)."""
    result = n
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def multiplicative_order(a: int, n: int) -> int:
    """Order of a in (Z/nZ)*. Returns 0 if gcd(a,n) > 1."""
    if math.gcd(a, n) > 1:
        return 0
    order, cur = 1, a % n
    while cur != 1:
        cur = (cur * a) % n
        order += 1
        if order > n:
            return 0
    return order


def primitive_root(p: int) -> Optional[int]:
    """Find the smallest primitive root mod prime p."""
    if p == 2:
        return 1
    phi = p - 1
    factors: set[int] = set()
    n = phi
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    return None


def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p) for odd prime p."""
    a = a % p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def discrete_log(x: int, g: int, p: int) -> Optional[int]:
    """Baby-step giant-step discrete log: find k such that g^k = x mod p."""
    if x % p == 0:
        return None
    x = x % p
    n = p - 1
    m = int(math.isqrt(n)) + 1
    table: dict[int, int] = {}
    power = 1
    for j in range(m):
        table[power] = j
        power = (power * g) % p
    factor = pow(g, n - m, p)
    gamma = x
    for i in range(m):
        if gamma in table:
            return (i * m + table[gamma]) % n
        gamma = (gamma * factor) % p
    return None


def build_character_table(p: int) -> Tuple[int, Dict[int, int], int]:
    """Build Dirichlet character table for prime p.

    Returns (g, log_table, phi) where g is a primitive root,
    log_table[x] = log_g(x) for x in 1..p-1, phi = p-1.
    Character chi_j(x) = exp(2*pi*i * j * log_table[x] / phi).
    """
    g = primitive_root(p)
    assert g is not None
    phi = p - 1
    log_table: dict[int, int] = {}
    power = 1
    for k in range(phi):
        log_table[power] = k
        power = (power * g) % p
    return g, log_table, phi


def poly_roots_mod(coeffs: Sequence[int], m: int) -> FrozenSet[int]:
    """All roots of polynomial (little-endian coeffs) in Z/mZ."""
    return frozenset(x for x in range(m) if eval_poly_mod(coeffs, x, m) == 0)
