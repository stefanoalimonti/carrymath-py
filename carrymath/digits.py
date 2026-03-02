"""Digit decomposition and polynomial evaluation."""

from __future__ import annotations

from typing import List, Sequence, Union

Numeric = Union[int, float, complex]


def to_digits(n: int, base: int = 2) -> List[int]:
    """Little-endian digit decomposition: n = sum(d[i] * base^i).

    >>> to_digits(13, 2)
    [1, 0, 1, 1]
    >>> to_digits(0)
    [0]
    """
    if n < 0:
        raise ValueError(f"to_digits requires n >= 0, got {n}")
    if n == 0:
        return [0]
    d: list[int] = []
    while n > 0:
        d.append(int(n % base))
        n //= base
    return d


def from_digits(digits: Sequence[int], base: int = 2) -> int:
    """Reconstruct integer from little-endian digits.

    >>> from_digits([1, 0, 1, 1], 2)
    13
    """
    n = 0
    for i, d in enumerate(digits):
        n += d * base**i
    return n


def to_bits_msb(n: int) -> List[int]:
    """Big-endian (MSB-first) binary digits.

    >>> to_bits_msb(13)
    [1, 1, 0, 1]
    """
    if n == 0:
        return [0]
    return [int(b) for b in bin(n)[2:]]


def eval_poly(coeffs: Sequence[Numeric], x: Numeric) -> Numeric:
    """Evaluate polynomial with little-endian coefficients at x.

    coeffs[i] is the coefficient of x^i.
    Uses Horner's method (reversed).

    >>> eval_poly([1, 2, 3], 10)  # 1 + 2*10 + 3*100
    321
    """
    result = coeffs[-1] if coeffs else 0
    for c in reversed(coeffs[:-1]):
        result = result * x + c
    return result


def eval_poly_mod(coeffs: Sequence[int], x: int, mod: int) -> int:
    """Evaluate polynomial mod m. Little-endian coefficients, exact int arithmetic.

    >>> eval_poly_mod([1, 2, 3], 10, 7)  # 321 mod 7
    6
    """
    val = 0
    xp = 1
    for c in coeffs:
        val = (val + c * xp) % mod
        xp = (xp * x) % mod
    return val
