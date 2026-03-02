"""Companion matrix, modular matrix, and spectral determinant.

The companion matrix M of the carry polynomial C(x) has eigenvalues that
encode the spectral structure of the carry chain. The spectral determinant
|det(I - M/l^s)| connects to the Euler product approximation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from carrymath.carry import carry_difference, carry_polynomial


def companion_matrix(coeffs: Sequence[int]) -> np.ndarray:
    """Build the companion matrix from polynomial coefficients.

    For a monic polynomial x^D + a_{D-1}x^{D-1} + ... + a_0,
    the companion matrix is D x D with the negated coefficients
    in the last column and 1's on the sub-diagonal.

    Parameters
    ----------
    coeffs : little-endian polynomial coefficients [a_0, a_1, ..., a_{D-1}, a_D]
        The leading coefficient a_D is used to normalize.

    Returns
    -------
    M : (D, D) numpy array
    """
    D = len(coeffs) - 1
    if D <= 0:
        return np.array([[0.0]])
    lead = coeffs[-1]
    M = np.zeros((D, D))
    for i in range(D):
        M[0, i] = -coeffs[D - 1 - i] / lead
    for i in range(1, D):
        M[i, i - 1] = 1.0
    return M


def companion_from_semiprime(
    p: int, q: int, base: int = 2, use_quotient: bool = True
) -> np.ndarray:
    """Build companion matrix directly from a semiprime factorization.

    Parameters
    ----------
    p, q : prime factors
    base : positional base
    use_quotient : if True, use Q(x) = C(x)/(x-base); if False, use
        the raw difference d(x) = G(x)H(x) - F(x).

    Returns
    -------
    M : companion matrix
    """
    if use_quotient:
        cp = carry_polynomial(p, q, base)
    else:
        cp = carry_difference(p, q, base)
    return companion_matrix(cp)


def spectral_det(M: np.ndarray, prime: int, s: complex, return_complex: bool = False) -> float:
    """Compute |det(I - M / prime^s)|.

    This is the per-prime spectral factor in the carry product
    approximation to zeta(s).

    Parameters
    ----------
    M : companion matrix
    prime : prime base (2, 3, 5, ...)
    s : complex parameter (typically on the critical line s = 1/2 + it)
    return_complex : if True, return the complex determinant instead of |det|

    Returns
    -------
    |det(I - M/prime^s)| or det(I - M/prime^s) if return_complex=True
    """
    D = M.shape[0]
    ps = complex(prime) ** s
    A = np.eye(D, dtype=np.complex128) - M / ps
    det = np.linalg.det(A)
    return det if return_complex else abs(det)


def carry_product(
    N: int,
    s: complex,
    primes: Sequence[int],
    base: int = 2,
    n_samples: int = 50,
) -> float:
    """Compute the averaged carry product Z_carry(s) = prod_l <|det(I - M_l/l^s)|>.

    Averages over `n_samples` random factorizations of semiprimes
    with the same bit length as N.

    Parameters
    ----------
    N : target integer (for bit-length reference)
    s : complex parameter
    primes : list of primes l to include in the product
    base : positional base
    n_samples : ensemble size for averaging

    Returns
    -------
    Z : carry product value
    """
    from carrymath.primes import random_semiprime_exact

    bits = N.bit_length()
    Z = 1.0
    for p_val in primes:
        det_sum = 0.0
        count = 0
        for _ in range(n_samples):
            result = random_semiprime_exact(bits)
            if result is None:
                continue
            _, p, q = result
            try:
                M = companion_from_semiprime(p, q, base)
                det_sum += spectral_det(M, p_val, s)
                count += 1
            except (np.linalg.LinAlgError, ValueError):
                continue
        if count > 0:
            Z *= det_sum / count
    return Z


def modular_matrix_row(N: int, row: int, width: Optional[int] = None) -> np.ndarray:
    """Row of the modular multiplication table.

    Returns the array [row*0 mod N, row*1 mod N, ..., row*(width-1) mod N].

    Parameters
    ----------
    N : the modulus
    row : which row (multiplier)
    width : number of columns (default: N)

    Returns
    -------
    result : 1D numpy array of length `width`
    """
    if width is None:
        width = N
    return np.array([(row * k) % N for k in range(width)])


def spectral_eigenvalues(p: int, q: int, base: int = 2, use_quotient: bool = True) -> np.ndarray:
    """One-shot: semiprime factorization to sorted eigenvalues.

    Parameters
    ----------
    p, q : prime factors
    base : positional base
    use_quotient : if True, use Q(x) = C(x)/(x-base)

    Returns
    -------
    eigenvalues : 1D complex array, sorted by descending magnitude
    """
    M = companion_from_semiprime(p, q, base, use_quotient)
    eigs = np.linalg.eigvals(M)
    return eigs[np.argsort(-np.abs(eigs))]


def carry_analysis(p: int, q: int, base: int = 2) -> dict:
    """Complete carry analysis bundle for a semiprime factorization.

    Returns
    -------
    dict with keys:
        'N': product
        'p', 'q': factors
        'g', 'h', 'f': digit sequences
        'carries': carry chain
        'convolutions': convolution sums
        'carry_poly': raw carry polynomial C(x)
        'quotient_poly': Q(x) = C(x)/(x-base)
        'companion': companion matrix of Q
        'eigenvalues': sorted eigenvalues
        'spectral_radius': max |eigenvalue|
        'trace_anomaly': penultimate carry value
    """
    from carrymath.carry import (
        carry_chain_with_conv,
        carry_difference,
    )
    from carrymath.digits import to_digits

    N = p * q
    g = to_digits(p, base)
    h = to_digits(q, base)
    f = to_digits(N, base)
    carries, convs = carry_chain_with_conv(g, h, base)
    cp = carry_difference(p, q, base)
    M = companion_from_semiprime(p, q, base)
    eigs = np.linalg.eigvals(M)
    eigs_sorted = eigs[np.argsort(-np.abs(eigs))]

    return {
        "N": N,
        "p": p,
        "q": q,
        "g": g,
        "h": h,
        "f": f,
        "carries": carries,
        "convolutions": convs,
        "carry_poly": cp,
        "quotient_poly": list(M[0, ::-1] * -M[0, 0] if M.shape[0] > 0 else []),
        "companion": M,
        "eigenvalues": eigs_sorted,
        "spectral_radius": float(np.max(np.abs(eigs_sorted))) if len(eigs_sorted) > 0 else 0.0,
        "trace_anomaly": carries[-2] if len(carries) >= 2 else 0,
    }
