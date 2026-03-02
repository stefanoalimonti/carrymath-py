"""
carrymath — Carry arithmetic framework for positional multiplication.

Provides tools for studying the carry chain arising from integer
multiplication in any base, including carry polynomials, companion
matrices, spectral determinants, exact enumeration, covariance
computation, and BFS entropy profiling.

Modules
-------
digits        Digit decomposition and polynomial evaluation
primes        Primality testing, prime/semiprime generation
carry         Carry chain, carry polynomial, convolution sums
matrix        Companion matrix, modular matrix, spectral determinant
enumerate     Exact enumeration over 2^{2K} bit configurations
covariance    Exact covariance computation (Lemma D verification)
entropy       BFS entropy curve for carry-chain factorization
number_theory Euler totient, primitive roots, Legendre symbol, characters
"""

__version__ = "0.1.0"

__all__ = [
    # digits
    "to_digits",
    "from_digits",
    "to_bits_msb",
    "eval_poly",
    "eval_poly_mod",
    # primes
    "is_prime",
    "sieve",
    "random_prime",
    "random_prime_range",
    "random_semiprime",
    "random_semiprime_exact",
    # carry
    "carry_chain",
    "carry_chain_with_conv",
    "carry_polynomial",
    "carry_polynomial_coeffs",
    "carry_difference",
    "quotient_polynomial",
    "convolution",
    "conv_at",
    "CarryChain",
    # matrix
    "companion_matrix",
    "companion_from_semiprime",
    "spectral_det",
    "carry_product",
    "modular_matrix_row",
    "spectral_eigenvalues",
    "carry_analysis",
    # covariance
    "bulk_covariance",
    "offdiagonal_covariance",
    "verify_lemma_d",
    "even_j_corrections",
    "induction_step",
    # entropy
    "bfs_states",
    "bfs_simple",
    "entropy_curve",
    "divisor_component",
    "phantom_counts",
    "entropy_profile",
    # enumerate
    "enumerate_exact",
    "enumerate_numpy",
    # number_theory
    "euler_totient",
    "multiplicative_order",
    "primitive_root",
    "legendre_symbol",
    "discrete_log",
    "build_character_table",
    "poly_roots_mod",
]

from carrymath.carry import (
    CarryChain,
    carry_chain,
    carry_chain_with_conv,
    carry_difference,
    carry_polynomial,
    carry_polynomial_coeffs,
    conv_at,
    convolution,
    quotient_polynomial,
)
from carrymath.covariance import (
    bulk_covariance,
    even_j_corrections,
    induction_step,
    offdiagonal_covariance,
    verify_lemma_d,
)
from carrymath.digits import eval_poly, eval_poly_mod, from_digits, to_bits_msb, to_digits
from carrymath.entropy import (
    bfs_simple,
    bfs_states,
    divisor_component,
    entropy_curve,
    entropy_profile,
    phantom_counts,
)
from carrymath.enumerate import (
    enumerate_exact,
    enumerate_numpy,
)
from carrymath.matrix import (
    carry_analysis,
    carry_product,
    companion_from_semiprime,
    companion_matrix,
    modular_matrix_row,
    spectral_det,
    spectral_eigenvalues,
)
from carrymath.number_theory import (
    build_character_table,
    discrete_log,
    euler_totient,
    legendre_symbol,
    multiplicative_order,
    poly_roots_mod,
    primitive_root,
)
from carrymath.primes import (
    is_prime,
    random_prime,
    random_prime_range,
    random_semiprime,
    random_semiprime_exact,
    sieve,
)
