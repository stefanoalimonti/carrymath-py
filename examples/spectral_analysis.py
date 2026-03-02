"""Spectral analysis: companion matrix eigenvalues and spectral determinant."""

import numpy as np

from carrymath.matrix import companion_from_semiprime, spectral_det
from carrymath.primes import random_semiprime, sieve

# Generate a semiprime and its companion matrix
N, p, q = random_semiprime(16)
M = companion_from_semiprime(p, q)
print(f"N = {N} = {p} * {q}")
print(f"Companion matrix shape: {M.shape}")
print(f"Eigenvalues: {np.linalg.eigvals(M)}")
print()

# Spectral determinant at first few primes
primes = sieve(20)
s = complex(0.5, 14.134725)  # Near first Riemann zero
print(f"Spectral determinants |det(I - M/l^s)| at s = {s}:")
for prime_l in primes:
    d = spectral_det(M, prime_l, s)
    euler = abs(1 - prime_l ** (-s)) ** (-1)
    ratio = d * abs(1 - prime_l ** (-s))
    print(f"  l={prime_l}: det={d:.6f}, 1/|1-l^{{-s}}|={euler:.6f}, ratio={ratio:.6f}")
