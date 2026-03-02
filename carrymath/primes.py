"""Primality testing, prime generation, and semiprime construction."""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple


def is_prime(n: int, rounds: int = 20) -> bool:
    """Miller-Rabin primality test.

    Probabilistic with error probability <= 4^{-rounds}. Uses random witnesses.
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for _ in range(rounds):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def sieve(limit: int) -> List[int]:
    """Sieve of Eratosthenes. Returns sorted list of primes up to limit.

    >>> sieve(20)
    [2, 3, 5, 7, 11, 13, 17, 19]
    """
    if limit < 2:
        return []
    is_p = bytearray(b"\x01") * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i * i :: i] = bytearray(len(is_p[i * i :: i]))
    return [i for i, v in enumerate(is_p) if v]


def random_prime(bits: int) -> int:
    """Generate a random prime with exactly `bits` bits."""
    while True:
        n = random.getrandbits(bits) | (1 << (bits - 1)) | 1
        if is_prime(n):
            return n


def random_prime_range(lo: int, hi: int) -> int:
    """Generate a random prime in [lo, hi)."""
    while True:
        n = random.randrange(lo | 1, hi, 2)
        if is_prime(n):
            return n


def random_semiprime(bits: int, balanced: bool = True) -> Tuple[int, int, int]:
    """Generate a random semiprime N = p*q with `bits` total bits.

    Parameters
    ----------
    bits : int
        Target bit length of N.
    balanced : bool
        If True, p and q have similar bit lengths (bits//2).
        If False, p and q can have different sizes.

    Returns
    -------
    (N, p, q) with p <= q.

    >>> N, p, q = random_semiprime(20)
    >>> N == p * q
    True
    """
    half = bits // 2
    if balanced:
        p = random_prime(half)
        q = random_prime(bits - half)
    else:
        p_bits = random.randint(max(2, bits // 4), bits * 3 // 4)
        p = random_prime(p_bits)
        q = random_prime(bits - p_bits)
    if p > q:
        p, q = q, p
    return p * q, p, q


def random_semiprime_exact(bits: int) -> Optional[Tuple[int, int, int]]:
    """Generate semiprime with EXACTLY `bits` bits. May return None if unlucky."""
    for _ in range(100):
        N, p, q = random_semiprime(bits)
        if N.bit_length() == bits:
            return N, p, q
    return None
