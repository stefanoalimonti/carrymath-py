"""Carry chain, carry polynomial, and convolution for positional multiplication.

The fundamental recurrence for base-b multiplication of G = g_0 g_1 ... g_{D-1}
and H = h_0 h_1 ... h_{D-1} is:

    conv_j = sum_{i=0}^{j} g_i * h_{j-i}
    total_j = conv_j + carry_j
    carry_{j+1} = floor(total_j / b)
    digit_j = total_j mod b

with carry_0 = 0 and g_0 = h_0 = 1 (LSB-first convention: g[0] and h[0] are the least significant digits).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from carrymath.digits import to_digits


def conv_at(g: Sequence[int], h: Sequence[int], j: int) -> int:
    """Compute convolution sum at position j: sum_{i=0}^{j} g_i * h_{j-i}.

    Handles boundary: terms with out-of-range index contribute 0.

    >>> conv_at([1, 0, 1], [1, 1, 0], 2)
    2
    """
    s = 0
    for i in range(j + 1):
        if i < len(g) and (j - i) < len(h):
            s += g[i] * h[j - i]
    return s


def convolution(g: Sequence[int], h: Sequence[int]) -> List[int]:
    """Full convolution of two digit sequences.

    Returns list of length len(g) + len(h) - 1.

    >>> convolution([1, 1], [1, 1])
    [1, 2, 1]
    """
    n = len(g) + len(h) - 1
    result = [0] * n
    for i, a in enumerate(g):
        for j, b in enumerate(h):
            result[i + j] += a * b
    return result


def carry_chain(
    g: Sequence[int],
    h: Sequence[int],
    base: int = 2,
    length: Optional[int] = None,
) -> List[int]:
    """Compute the full carry chain for positional multiplication.

    Parameters
    ----------
    g, h : digit sequences (MSB-first model: g[0]=h[0]=1 typically)
    base : multiplication base
    length : number of positions to compute (default: len(g) + len(h))

    Returns
    -------
    carries : list of length `length`+1 where carries[0] = 0.

    >>> carry_chain([1, 1, 0, 1], [1, 0, 1, 1])  # 13 * 11 in base 2
    [0, 0, 1, 1, 2, 2, 1, 0]
    """
    if length is None:
        length = len(g) + len(h) - 1
    carries = [0]
    for j in range(length):
        cv = conv_at(g, h, j)
        total = cv + carries[-1]
        carries.append(total // base)
    return carries


def carry_chain_with_conv(
    g: Sequence[int],
    h: Sequence[int],
    base: int = 2,
    length: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Compute carry chain AND convolution sums simultaneously.

    Returns (carries, convolutions) where both are aligned by position.
    """
    if length is None:
        length = len(g) + len(h) - 1
    carries = [0]
    convs = []
    for j in range(length):
        cv = conv_at(g, h, j)
        convs.append(cv)
        total = cv + carries[-1]
        carries.append(total // base)
    return carries, convs


def carry_difference(p: int, q: int, base: int = 2) -> List[int]:
    """Compute d(x) = G(x)*H(x) - F(x) = (x - base) * C(x).

    Returns the raw difference polynomial (little-endian), which equals
    (x - base) times the carry polynomial C(x).
    """
    n = p * q
    gd = to_digits(p, base)
    hd = to_digits(q, base)
    fd = to_digits(n, base)
    conv = convolution(gd, hd)
    mx = max(len(conv), len(fd))
    diff = []
    for i in range(mx):
        gi = conv[i] if i < len(conv) else 0
        fi = fd[i] if i < len(fd) else 0
        diff.append(gi - fi)
    while len(diff) > 1 and diff[-1] == 0:
        diff.pop()
    return diff


def carry_polynomial(p: int, q: int, base: int = 2) -> List[int]:
    """Compute carry polynomial C(x) coefficients (little-endian).

    From the Carry Representation Theorem (Diaconis-Fulman 2009):
        P(x) * Q(x) = F(x) + (x - base) * C(x)
    where P, Q, F are digit polynomials of p, q, p*q.

    Returns C(x) after dividing out the (x - base) factor.
    """
    diff = carry_difference(p, q, base)
    return quotient_polynomial(diff, base)


def carry_polynomial_coeffs(p: int, q: int, base: int = 2) -> List[int]:
    """Carry polynomial as explicit carry values c_0, c_1, ..., c_D.

    These are the actual carry values at each position, computed from
    the digit recurrence. carry[0] = 0 always.
    """
    gd = to_digits(p, base)
    hd = to_digits(q, base)
    return carry_chain(gd, hd, base)


def quotient_polynomial(carry_poly: Sequence[int], base: int = 2) -> List[int]:
    """Q(x) = C(x) / (x - base) via synthetic division.

    The carry polynomial C(x) is always divisible by (x - base).
    Returns little-endian coefficients of Q(x).
    """
    n = len(carry_poly)
    if n <= 1:
        return [0]
    q = [0] * (n - 1)
    q[-1] = carry_poly[-1]
    for i in range(n - 2, 0, -1):
        q[i - 1] = carry_poly[i] + base * q[i]
    remainder = carry_poly[0] + base * q[0]
    if remainder != 0:
        raise ValueError(f"Polynomial not divisible by (x - {base}); remainder = {remainder}")
    return q


class CarryChain:
    """Object-oriented carry chain for a digit pair (g, h) in base b.

    Provides lazy computation and caching of carries, convolutions,
    and derived quantities.

    Parameters
    ----------
    g : list of ints — digits of first factor (any endianness convention)
    h : list of ints — digits of second factor
    base : int — positional base (default 2)
    """

    def __init__(
        self,
        g: Sequence[int],
        h: Sequence[int],
        base: int = 2,
    ):
        self.g = list(g)
        self.h = list(h)
        self.base = base
        self._carries: Optional[List[int]] = None
        self._convs: Optional[List[int]] = None

    def _compute(self) -> None:
        if self._carries is not None:
            return
        self._carries, self._convs = carry_chain_with_conv(self.g, self.h, self.base)

    @property
    def carries(self) -> List[int]:
        """Carry values [c_0=0, c_1, ..., c_D]."""
        self._compute()
        assert self._carries is not None
        return self._carries

    @property
    def convolutions(self) -> List[int]:
        """Convolution sums [conv_0, conv_1, ..., conv_{D-1}]."""
        self._compute()
        assert self._convs is not None
        return self._convs

    @property
    def D(self) -> int:
        """Number of digit positions."""
        return len(self.g) + len(self.h) - 1

    @property
    def total(self) -> List[int]:
        """Total at each position: total_j = conv_j + carry_j."""
        return [c + v for c, v in zip(self.carries, self.convolutions)]

    @property
    def product_digits(self) -> List[int]:
        """Output digits of the product: total_j mod base, plus final carry."""
        digits = [t % self.base for t in self.total]
        final = self.carries[-1]
        while final > 0:
            digits.append(final % self.base)
            final //= self.base
        return digits

    def trace_anomaly(self) -> int:
        """The MSB carry value c_{D-1} (trace anomaly observable)."""
        return self.carries[-2] if len(self.carries) >= 2 else 0

    @classmethod
    def from_integers(cls, p: int, q: int, base: int = 2) -> "CarryChain":
        """Create CarryChain from two integers."""
        return cls(to_digits(p, base), to_digits(q, base), base)

    @classmethod
    def from_bits_msb(cls, g_bits: Sequence[int], h_bits: Sequence[int]) -> "CarryChain":
        """Create CarryChain from MSB-first bit sequences (g[0]=h[0]=1)."""
        return cls(list(g_bits), list(h_bits), base=2)

    def __repr__(self) -> str:
        return f"CarryChain(g={self.g}, h={self.h}, base={self.base}, D={self.D})"
