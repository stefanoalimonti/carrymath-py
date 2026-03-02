"""Entropy profiling of a semiprime: S0, tau, phantom decomposition."""

from carrymath.entropy import entropy_profile
from carrymath.primes import random_semiprime_exact

N_result = random_semiprime_exact(20)
if N_result:
    N, p, q = N_result
    print(f"N = {N} = {p} * {q}  ({N.bit_length()} bits)")
    print()

    prof = entropy_profile(N, max_pos=min(12, N.bit_length()))
    print(f"{'k':>3} {'S0':>8} {'tau':>8} {'Phantom':>8} {'H(k)':>8} {'Phi/S0':>8}")
    print("-" * 50)
    for k in range(len(prof['S0'])):
        print(f"{k+1:>3} {prof['S0'][k]:>8} {prof['tau'][k]:>8} "
              f"{prof['phantom'][k]:>8} {prof['H'][k]:>8.2f} {prof['phantom_ratio'][k]:>8.3f}")
