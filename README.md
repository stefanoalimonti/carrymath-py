# carrymath

A Python library for studying the carry chain arising from positional integer multiplication.

Provides tools for carry polynomials, companion matrices, spectral determinants, exact enumeration, covariance computation, and BFS entropy profiling — the mathematical objects from the [carry arithmetic research program](https://github.com/stefanoalimonti/carry-arithmetic).

## Installation

```bash
pip install -e .

# With optional dependencies (mpmath for high precision, scipy for advanced numerics):
pip install -e ".[all]"
```

## Quick Start

```python
from carrymath import CarryChain, carry_polynomial

# Carry chain for 13 × 11 = 143 (base 2, little-endian digits)
cc = CarryChain.from_integers(13, 11)
print(cc.carries)       # [0, 0, 0, 0, 1, 1, 1, 1]
print(cc.convolutions)  # [1, 1, 1, 3, 1, 1, 1]
print(cc.trace_anomaly())  # 1

# Carry polynomial: P(x)Q(x) = F(x) + (x-2)C(x)
C = carry_polynomial(13, 11)
print(C)  # [0, 0, 0, -1, -1, -1, -1]

# Exact covariance verification (Lemma D)
from carrymath.covariance import bulk_covariance
from fractions import Fraction
r = bulk_covariance(6)
print(r['cov'][3])  # 1/4  — Cov(carry_3, conv_3) = (3-1)/8
print(r['cov'][5])  # 1/2  — Cov(carry_5, conv_5) = (5-1)/8
```

## Modules

### `carrymath.digits`
Digit decomposition and polynomial evaluation.

| Function | Description |
|----------|-------------|
| `to_digits(n, base=2)` | Little-endian digit decomposition |
| `from_digits(digits, base=2)` | Reconstruct integer from digits |
| `to_bits_msb(n)` | MSB-first binary digits |
| `eval_poly(coeffs, x)` | Evaluate polynomial (Horner's method) |
| `eval_poly_mod(coeffs, x, mod)` | Evaluate polynomial mod m |

### `carrymath.primes`
Primality testing and prime/semiprime generation.

| Function | Description |
|----------|-------------|
| `is_prime(n, rounds=20)` | Miller-Rabin primality test (probabilistic) |
| `sieve(limit)` | Sieve of Eratosthenes |
| `random_prime(bits)` | Random prime with given bit length |
| `random_prime_range(lo, hi)` | Random prime in range |
| `random_semiprime(bits)` | Random semiprime N=pq |
| `random_semiprime_exact(bits)` | Semiprime with exact bit count |

### `carrymath.carry`
Core carry chain computation.

| Function/Class | Description |
|----------------|-------------|
| `carry_chain(g, h, base=2)` | Compute full carry chain |
| `carry_chain_with_conv(g, h)` | Carries + convolution sums |
| `carry_polynomial(p, q, base=2)` | C(x) via CRT: P(x)Q(x) = F(x) + (x-b)C(x) |
| `carry_difference(p, q, base=2)` | Raw difference d(x) = (x-b)C(x) |
| `quotient_polynomial(cp, base=2)` | Synthetic division by (x-b) |
| `convolution(g, h)` | Full polynomial convolution |
| `carry_polynomial_coeffs(p, q)` | Carry values c_0, c_1, ..., c_D |
| `conv_at(g, h, j)` | Single-position convolution sum |
| `CarryChain` | OOP carry chain with lazy caching |

### `carrymath.matrix`
Companion matrix and spectral analysis.

| Function | Description |
|----------|-------------|
| `companion_matrix(coeffs)` | Build companion matrix from polynomial |
| `companion_from_semiprime(p, q)` | Companion matrix for a factorization |
| `spectral_det(M, l, s)` | \|det(I - M/l^s)\| |
| `carry_product(N, s, primes)` | Averaged carry product Z_carry(s) |
| `modular_matrix_row(N, row)` | Row of the modular multiplication table |
| `spectral_eigenvalues(p, q)` | One-shot: semiprime to sorted eigenvalues |
| `carry_analysis(p, q)` | Full analysis bundle (digits, carries, eigenvalues, etc.) |

### `carrymath.enumerate`
Exact enumeration over 2^{2K} bit configurations.

| Function | Description |
|----------|-------------|
| `enumerate_exact(K)` | Exact `Fraction` expectations at window K |
| `enumerate_numpy(K)` | Float-based enumeration with covariance |

### `carrymath.covariance`
Exact covariance computation and Lemma D verification.

| Function | Description |
|----------|-------------|
| `bulk_covariance(K)` | Cov(carry_j, conv_j) for all j |
| `offdiagonal_covariance(K)` | Cov(carry_j, g_i h_{j-i}) per-pair |
| `verify_lemma_d(K)` | Full Lemma D verification |
| `even_j_corrections(K)` | ε_j for even j |
| `induction_step(K)` | ΔCov = Cov(j+2) - Cov(j) |

### `carrymath.entropy`
BFS entropy profiling for carry-chain factorization.

| Function | Description |
|----------|-------------|
| `bfs_states(N, max_pos)` | Count valid (p,q,carry) states at each position |
| `bfs_simple(N, max_pos)` | Simplified BFS via modular constraint |
| `entropy_curve(N)` | H(k) = log₂(S₀(k)) |
| `divisor_component(N)` | τ(M_k) genuine factor pairs |
| `phantom_counts(N)` | Φ(k) = S₀(k) - τ(M_k) |
| `entropy_profile(N)` | Full decomposition: S₀, τ, Φ, H |

### `carrymath.number_theory`
Classical number-theoretic utilities.

| Function | Description |
|----------|-------------|
| `euler_totient(n)` | Euler's φ(n) |
| `multiplicative_order(a, n)` | Order of a in (Z/nZ)* |
| `primitive_root(p)` | Smallest primitive root mod p |
| `legendre_symbol(a, p)` | Legendre symbol (a/p) |
| `discrete_log(x, g, p)` | Baby-step giant-step |
| `build_character_table(p)` | Dirichlet character table |
| `poly_roots_mod(coeffs, m)` | Polynomial roots in Z/mZ |

## Examples

```bash
python examples/basic_carry_chain.py       # Core carry chain demo
python examples/covariance_verification.py  # Exact Lemma D verification
python examples/entropy_profile.py          # BFS entropy decomposition
python examples/spectral_analysis.py        # Companion matrix eigenvalues
```

## Tests

```bash
pytest tests/ -v
```

301 tests covering all modules (digits, primes, carry, matrix, enumerate, covariance, entropy, number_theory) plus cross-validation vectors shared with the JavaScript and C implementations via `tests/data/test_vectors.json`.

## Key Mathematical Results

This library implements verification of several proved results from the carry arithmetic series:

- **Lemma A**: E[conv_j] = (j+3)/4
- **Parity Lemma**: P(conv_j + carry_j odd) = 1/2
- **Theorem 3**: E[carry_j] = (j-1)/4
- **Lemma D**: Cov(carry_j, g_i h_{j-i}) = 1/8 for off-diagonal i ≠ j-i
- **Main Theorem**: Cov(carry_j, conv_j) = (j-1)/8 for odd j ≥ 3
- **CRT**: P(x)Q(x) = F(x) + (x-b)C(x) (Carry Representation Theorem)

## Feature Comparison

| Module | Python | JavaScript | C |
|--------|--------|------------|---|
| digits | ✅ (5 fn) | ✅ (6 fn) | ✅ (2 fn) |
| primes | ✅ (6 fn) | ✅ (6 fn) | ✅ (3 fn) |
| carry | ✅ (9 fn) | ✅ (9 fn) | ✅ (3 fn) |
| matrix | ✅ (7 fn) | ✅ (7 fn) | — |
| enumerate | ✅ (2 fn) | ✅ (2 fn) | ✅ (1 fn) |
| covariance | ✅ (5 fn) | ✅ (5 fn) | — |
| entropy | ✅ (6 fn) | ✅ (6 fn) | ✅ (5 fn) |
| number_theory | ✅ (7 fn) | ✅ (7 fn) | ✅ (4 fn) |

## Requirements

- Python 3.8+
- NumPy ≥ 1.20
- Optional: mpmath (high precision), SciPy (advanced numerics)

## Precision Notes

- All core functions (`digits`, `primes`, `carry`, `number_theory`, `entropy`) use Python's native arbitrary-precision `int` — no overflow for any input size.
- `companion_matrix`, `spectral_eigenvalues`, and `enumerate_numpy` use NumPy float64 arrays; coefficients with more than ~15 significant digits lose precision. For exact results, use `enumerate_exact` (rational `Fraction` arithmetic).

## Author

**Stefano Alimonti** · [ORCID 0009-0009-1183-1698](https://orcid.org/0009-0009-1183-1698) — [carry arithmetic research program](https://github.com/stefanoalimonti/carry-arithmetic)

## References

This library implements and verifies results from the carry arithmetic paper series:

- **Paper A** — *Spectral Theory of Carry Propagation*: Diaconis-Fulman eigenvalues, Gershgorin bounds, CRT identity
- **Paper B** — *Carry Zeta Approximation*: spectral determinant framework for the Euler product
- **Paper C** — *Matrix Statistics (GOE/GUE)*: random matrix theory for companion matrices
- **Paper D** — *Entropy Bound / Factorization*: BFS entropy, phantom halo, divisor bounds
- **Paper E** — *Trace Anomaly*: resolvent universality, π emergence mechanism
- **Paper F** — *Covariance Structure*: exact Cov(carry_j, conv_j), Lemma D
- **Paper Frobenius** — *Witt Vectors and Carry Arithmetic*: algebraic lifting to W_n(F_p)

See the [carry-arithmetic repository](https://github.com/stefanoalimonti/carry-arithmetic) for the full paper collection.

### Citation

```bibtex
@software{alimonti2026carrymath,
  author  = {Alimonti, Stefano},
  title   = {carrymath: A library for carry chain analysis in positional multiplication},
  year    = {2026},
  url     = {https://github.com/stefanoalimonti/carrymath-py}
}
```

## License

MIT — see [LICENSE](LICENSE).
