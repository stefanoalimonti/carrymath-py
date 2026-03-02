"""Comprehensive pytest tests for all carrymath modules.

Covers: digits, primes, matrix, entropy, enumerate, number_theory, covariance.
"""

from fractions import Fraction

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# digits.py
# ---------------------------------------------------------------------------
from carrymath.digits import eval_poly, eval_poly_mod, from_digits, to_bits_msb, to_digits


class TestToDigits:
    @pytest.mark.parametrize(
        "n, base, expected",
        [
            (0, 2, [0]),
            (0, 10, [0]),
            (1, 2, [1]),
            (13, 2, [1, 0, 1, 1]),
            (255, 2, [1, 1, 1, 1, 1, 1, 1, 1]),
            (10, 10, [0, 1]),
            (123, 10, [3, 2, 1]),
            (7, 3, [1, 2]),
            (8, 16, [8]),
        ],
    )
    def test_known_values(self, n, base, expected):
        assert to_digits(n, base) == expected

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="n >= 0"):
            to_digits(-1, 2)

    @pytest.mark.parametrize("base", [2, 3, 10, 16])
    def test_zero_is_single_digit(self, base):
        assert to_digits(0, base) == [0]


class TestFromDigits:
    @pytest.mark.parametrize(
        "n, base",
        [
            (0, 2),
            (1, 2),
            (13, 2),
            (255, 2),
            (1024, 2),
            (0, 10),
            (42, 10),
            (999, 10),
            (100, 3),
            (100, 16),
        ],
    )
    def test_roundtrip(self, n, base):
        assert from_digits(to_digits(n, base), base) == n

    def test_explicit(self):
        assert from_digits([1, 0, 1, 1], 2) == 13
        assert from_digits([3, 2, 1], 10) == 123


class TestToBitsMsb:
    @pytest.mark.parametrize(
        "n, expected",
        [
            (0, [0]),
            (1, [1]),
            (13, [1, 1, 0, 1]),
            (255, [1, 1, 1, 1, 1, 1, 1, 1]),
            (8, [1, 0, 0, 0]),
        ],
    )
    def test_known(self, n, expected):
        assert to_bits_msb(n) == expected

    def test_msb_is_one_for_positive(self):
        for n in [1, 2, 5, 17, 1000]:
            assert to_bits_msb(n)[0] == 1


class TestEvalPoly:
    @pytest.mark.parametrize(
        "coeffs, x, expected",
        [
            ([1, 2, 3], 10, 321),  # 1 + 20 + 300
            ([1, 2, 3], 0, 1),  # constant term
            ([5], 100, 5),  # constant poly
            ([0, 0, 1], 3, 9),  # x^2 at x=3
            ([1, 1, 1, 1], 2, 15),  # 1+2+4+8
        ],
    )
    def test_known_values(self, coeffs, x, expected):
        assert eval_poly(coeffs, x) == expected

    def test_empty_coeffs(self):
        assert eval_poly([], 10) == 0

    def test_digit_polynomial_identity(self):
        """eval_poly(to_digits(n, b), b) == n"""
        for n in [0, 1, 42, 1023]:
            for b in [2, 10]:
                assert eval_poly(to_digits(n, b), b) == n


class TestEvalPolyMod:
    @pytest.mark.parametrize(
        "coeffs, x, mod, expected",
        [
            ([1, 2, 3], 10, 7, 321 % 7),
            ([1, 0, 1], 2, 5, 5 % 5),  # 1 + 0 + 4 = 5 mod 5 = 0
            ([1, 1, 1], 3, 100, 13),  # 1+3+9 = 13
        ],
    )
    def test_known_values(self, coeffs, x, mod, expected):
        assert eval_poly_mod(coeffs, x, mod) == expected

    def test_agrees_with_eval_poly(self):
        coeffs = [3, 1, 4, 1, 5]
        for x in [2, 7, 13]:
            for mod in [3, 7, 11, 100]:
                assert eval_poly_mod(coeffs, x, mod) == eval_poly(coeffs, x) % mod

    def test_empty_coeffs(self):
        assert eval_poly_mod([], 10, 7) == 0


# ---------------------------------------------------------------------------
# primes.py
# ---------------------------------------------------------------------------
from carrymath.primes import is_prime, random_prime, random_semiprime, random_semiprime_exact, sieve


class TestIsPrime:
    @pytest.mark.parametrize(
        "n, expected",
        [
            (0, False),
            (1, False),
            (2, True),
            (3, True),
            (4, False),
            (5, True),
            (6, False),
            (7, True),
            (9, False),
            (11, True),
            (15, False),
            (17, True),
            (49, False),
            (97, True),
            (561, False),  # Carmichael number
            (1009, True),
        ],
    )
    def test_known(self, n, expected):
        assert is_prime(n) == expected, f"is_prime({n}) should be {expected}"


class TestSieve:
    def test_sieve_20(self):
        assert sieve(20) == [2, 3, 5, 7, 11, 13, 17, 19]

    def test_sieve_1(self):
        assert sieve(1) == []

    def test_sieve_0(self):
        assert sieve(0) == []

    def test_sieve_2(self):
        assert sieve(2) == [2]

    def test_sieve_large(self):
        primes = sieve(100)
        assert len(primes) == 25
        assert primes[0] == 2
        assert primes[-1] == 97


class TestRandomPrime:
    @pytest.mark.parametrize("bits", [8, 10, 16, 20])
    def test_correct_bit_length_and_primality(self, bits):
        p = random_prime(bits)
        assert p.bit_length() == bits, f"Expected {bits}-bit prime, got {p.bit_length()}-bit"
        assert is_prime(p), f"{p} should be prime"


class TestRandomSemiprime:
    @pytest.mark.parametrize("bits", [16, 20, 24])
    def test_factors_correctly(self, bits):
        N, p, q = random_semiprime(bits)
        assert N == p * q
        assert p <= q
        assert is_prime(p)
        assert is_prime(q)


class TestRandomSemiprimeExact:
    @pytest.mark.parametrize("bits", [16, 20])
    def test_exact_bit_length(self, bits):
        result = random_semiprime_exact(bits)
        if result is not None:
            N, p, q = result
            assert N.bit_length() == bits
            assert N == p * q
            assert is_prime(p)
            assert is_prime(q)


# ---------------------------------------------------------------------------
# matrix.py
# ---------------------------------------------------------------------------
from carrymath.matrix import (
    carry_product,
    companion_from_semiprime,
    companion_matrix,
    modular_matrix_row,
    spectral_det,
)


class TestCompanionMatrix:
    def test_shape(self):
        coeffs = [1, 2, 3, 1]  # degree 3
        M = companion_matrix(coeffs)
        assert M.shape == (3, 3)

    def test_degenerate(self):
        M = companion_matrix([5])
        assert M.shape == (1, 1)

    def test_subdiagonal_ones(self):
        M = companion_matrix([2, -3, 1, 1])  # degree 3
        for i in range(1, M.shape[0]):
            assert M[i, i - 1] == 1.0


class TestCompanionFromSemiprime:
    @pytest.mark.parametrize("p, q", [(3, 5), (13, 17)])
    def test_runs_and_shape_quotient(self, p, q):
        M = companion_from_semiprime(p, q)
        assert M.ndim == 2
        assert M.shape[0] == M.shape[1]

    @pytest.mark.parametrize("p, q", [(3, 5), (7, 11), (13, 17)])
    def test_runs_without_quotient(self, p, q):
        M = companion_from_semiprime(p, q, use_quotient=False)
        assert M.ndim == 2
        assert M.shape[0] == M.shape[1]


class TestSpectralDet:
    def test_identity_when_M_zero(self):
        M = np.zeros((3, 3))
        result = spectral_det(M, 2, 0.5 + 14.134j)
        assert abs(result - 1.0) < 1e-10, "det(I - 0) should be 1"

    def test_return_complex(self):
        M = np.zeros((2, 2))
        det = spectral_det(M, 2, 1.0, return_complex=True)
        assert isinstance(det, (complex, np.complexfloating))

    def test_positive_for_small_semiprime(self):
        M = companion_from_semiprime(3, 5, use_quotient=False)
        val = spectral_det(M, 2, 0.5 + 14.134j)
        assert val >= 0, "|det| should be non-negative"


class TestModularMatrixRow:
    def test_basic(self):
        row = modular_matrix_row(7, 3)
        expected = np.array([(3 * k) % 7 for k in range(7)])
        np.testing.assert_array_equal(row, expected)

    def test_first_element_zero(self):
        row = modular_matrix_row(10, 5)
        assert row[0] == 0

    def test_custom_width(self):
        row = modular_matrix_row(7, 3, width=4)
        assert len(row) == 4
        np.testing.assert_array_equal(row, np.array([0, 3, 6, 2]))


class TestCarryProduct:
    def test_runs_for_small_input(self):
        val = carry_product(15, 0.5 + 14.134j, [2, 3], n_samples=5)
        assert isinstance(val, float)
        assert val > 0


# ---------------------------------------------------------------------------
# entropy.py
# ---------------------------------------------------------------------------
from carrymath.entropy import bfs_simple, entropy_curve, entropy_profile


class TestBfsSimple:
    def test_N15(self):
        counts = bfs_simple(15, max_pos=4)
        assert len(counts) == 4
        assert all(c >= 1 for c in counts)

    def test_counts_decrease_or_equal(self):
        """For a small semiprime, the count at position 1 should be positive."""
        counts = bfs_simple(21, max_pos=5)
        assert counts[0] >= 1


class TestEntropyCurve:
    def test_returns_floats(self):
        H = entropy_curve(15, max_pos=4)
        assert isinstance(H, list)
        assert all(isinstance(h, float) for h in H)

    def test_non_negative(self):
        H = entropy_curve(21, max_pos=5)
        assert all(h >= 0 for h in H)


class TestEntropyProfile:
    def test_correct_keys(self):
        prof = entropy_profile(15, max_pos=4)
        for key in ["S0", "tau", "phantom", "H", "phantom_ratio"]:
            assert key in prof, f"Missing key '{key}'"

    def test_lengths_consistent(self):
        prof = entropy_profile(21, max_pos=5)
        n = len(prof["S0"])
        assert len(prof["tau"]) == n
        assert len(prof["phantom"]) == n
        assert len(prof["H"]) == n
        assert len(prof["phantom_ratio"]) == n

    def test_phantom_equals_S0_minus_tau(self):
        prof = entropy_profile(15, max_pos=4)
        for s, t, p in zip(prof["S0"], prof["tau"], prof["phantom"]):
            assert p == s - t


# ---------------------------------------------------------------------------
# enumerate.py
# ---------------------------------------------------------------------------
from carrymath.enumerate import enumerate_exact, enumerate_numpy


class TestEnumerateExact:
    def test_K3_expected_carry(self):
        r = enumerate_exact(3)
        for j in range(1, r["D"] + 1):
            if j < len(r["E_carry"]):
                expected = Fraction(j - 1, 4)
                assert r["E_carry"][j] == expected, (
                    f"E[carry_{j}] = {r['E_carry'][j]}, expected {expected}"
                )

    @pytest.mark.parametrize("K", [2, 3, 4])
    def test_carry_0_is_zero(self, K):
        r = enumerate_exact(K)
        assert r["E_carry"][0] == Fraction(0)

    def test_c1_at_K3(self):
        r = enumerate_exact(3)
        # c1 = E_carry[D-1] = E_carry[3] = (3-1)/4 = 1/2
        assert r["c1"] == Fraction(1, 2), f"c1 = {r['c1']}, expected 1/2"

    def test_keys(self):
        r = enumerate_exact(2)
        for key in ["E_carry", "E_conv", "c1", "K", "D"]:
            assert key in r


class TestEnumerateNumpy:
    @pytest.mark.parametrize("K", [2, 3, 4])
    def test_carries_close_to_exact(self, K):
        exact = enumerate_exact(K)
        approx = enumerate_numpy(K)
        for j in range(len(exact["E_carry"])):
            expected = float(exact["E_carry"][j])
            assert abs(approx["carries"][j] - expected) < 1e-12, (
                f"K={K}, j={j}: numpy={approx['carries'][j]}, exact={expected}"
            )

    def test_c1_close_to_exact(self):
        exact = enumerate_exact(3)
        approx = enumerate_numpy(3)
        assert abs(approx["c1"] - float(exact["c1"])) < 1e-12

    def test_has_variance_keys(self):
        r = enumerate_numpy(3)
        assert "var_carry" in r
        assert "var_conv" in r
        assert "cov_carry_conv" in r


# ---------------------------------------------------------------------------
# number_theory.py
# ---------------------------------------------------------------------------
from carrymath.number_theory import (
    discrete_log,
    euler_totient,
    legendre_symbol,
    multiplicative_order,
    poly_roots_mod,
    primitive_root,
)


class TestEulerTotient:
    @pytest.mark.parametrize(
        "n, expected",
        [
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 2),
            (6, 2),
            (7, 6),
            (8, 4),
            (12, 4),
            (15, 8),
            (100, 40),
        ],
    )
    def test_known(self, n, expected):
        assert euler_totient(n) == expected, f"phi({n}) = {euler_totient(n)}, expected {expected}"


class TestMultiplicativeOrder:
    @pytest.mark.parametrize(
        "a, n, expected",
        [
            (3, 7, 6),
            (2, 7, 3),
            (2, 5, 4),
            (3, 10, 4),
        ],
    )
    def test_known(self, a, n, expected):
        assert multiplicative_order(a, n) == expected

    def test_not_coprime(self):
        assert multiplicative_order(2, 4) == 0


class TestPrimitiveRoot:
    @pytest.mark.parametrize(
        "p, expected",
        [
            (2, 1),
            (7, 3),
            (11, 2),
            (13, 2),
        ],
    )
    def test_known(self, p, expected):
        assert primitive_root(p) == expected

    def test_is_generator(self):
        for p in [5, 7, 11, 13, 17]:
            g = primitive_root(p)
            assert g is not None
            assert multiplicative_order(g, p) == p - 1


class TestLegendreSymbol:
    @pytest.mark.parametrize(
        "a, p, expected",
        [
            (1, 7, 1),  # 1 is always a QR
            (2, 7, 1),  # 2 is a QR mod 7 (3^2 = 2 mod 7)
            (3, 7, -1),  # 3 is a QNR mod 7
            (0, 5, 0),  # 0 mod p
            (4, 5, 1),  # 4 = 2^2
            (2, 5, -1),  # 2 is QNR mod 5
        ],
    )
    def test_known(self, a, p, expected):
        assert legendre_symbol(a, p) == expected


class TestDiscreteLog:
    def test_small_case(self):
        # 3 is a primitive root mod 7: 3^1=3, 3^2=2, 3^3=6, 3^4=4, 3^5=5, 3^6=1
        k = discrete_log(2, 3, 7)
        assert k is not None
        assert pow(3, k, 7) == 2

    def test_identity(self):
        k = discrete_log(1, 3, 7)
        assert k is not None
        assert pow(3, k, 7) == 1

    @pytest.mark.parametrize("x", [1, 2, 3, 4, 5, 6])
    def test_all_elements_mod7(self, x):
        k = discrete_log(x, 3, 7)
        assert k is not None
        assert pow(3, k, 7) == x

    def test_zero_returns_none(self):
        assert discrete_log(0, 3, 7) is None


class TestPolyRootsMod:
    def test_x2_minus_1_mod_8(self):
        # x^2 - 1 mod 8 : coeffs = [-1, 0, 1] (little-endian)
        roots = poly_roots_mod([-1, 0, 1], 8)
        assert roots == frozenset({1, 3, 5, 7})

    def test_x_mod_5(self):
        # x mod 5 : coeffs = [0, 1]
        roots = poly_roots_mod([0, 1], 5)
        assert roots == frozenset({0})

    def test_constant_nonzero(self):
        # 3 mod 7 has no roots
        roots = poly_roots_mod([3], 7)
        assert roots == frozenset()

    def test_x2_mod_4(self):
        # x^2 mod 4 : roots are 0 and 2
        roots = poly_roots_mod([0, 0, 1], 4)
        assert roots == frozenset({0, 2})


# ---------------------------------------------------------------------------
# covariance.py — additional tests beyond test_carry.py
# ---------------------------------------------------------------------------
from carrymath.covariance import (
    bulk_covariance,
    even_j_corrections,
    induction_step,
    verify_lemma_d,
)


class TestEvenJCorrections:
    @pytest.mark.parametrize("K", [4, 6])
    def test_corrections_are_fractions(self, K):
        corrections = even_j_corrections(K)
        for j, eps in corrections.items():
            assert j % 2 == 0, f"Key {j} should be even"
            assert isinstance(eps, Fraction)

    def test_K4_has_even_entries(self):
        corrections = even_j_corrections(4)
        assert len(corrections) > 0
        assert 2 in corrections or 4 in corrections


class TestInductionStep:
    def test_odd_to_odd_is_quarter(self):
        deltas = induction_step(8)
        for j in range(1, 8 - 2, 2):  # odd j where j+2 is also odd
            assert deltas[j] == Fraction(1, 4), f"Delta at j={j}: {deltas[j]}, expected 1/4"

    @pytest.mark.parametrize("K", [4, 6])
    def test_returns_fractions(self, K):
        deltas = induction_step(K)
        for v in deltas.values():
            assert isinstance(v, Fraction)


class TestBulkCovarianceExtra:
    def test_odd_formula(self):
        r = bulk_covariance(8)
        for j in range(3, r["D"], 2):
            expected = Fraction(j - 1, 8)
            assert r["cov"][j] == expected

    def test_formula_check_all_match(self):
        r = bulk_covariance(6)
        for j, info in r["formula_check"].items():
            assert info["match"], f"formula_check failed at j={j}"


class TestVerifyLemmaDExtra:
    @pytest.mark.parametrize("K", [3, 4, 5])
    def test_holds(self, K):
        result = verify_lemma_d(K)
        assert result["lemma_d_holds"], f"Lemma D failed at K={K}"
        assert result["n_interior_offdiag"] == result["n_interior_offdiag_match"]


class TestSpectralEigenvalues:
    def test_returns_sorted_by_magnitude(self):
        from carrymath import spectral_eigenvalues

        eigs = spectral_eigenvalues(7, 11, 2)
        mags = [abs(e) for e in eigs]
        assert mags == sorted(mags, reverse=True)

    def test_correct_count(self):
        from carrymath import spectral_eigenvalues

        eigs = spectral_eigenvalues(7, 11, 2)
        assert len(eigs) > 0


class TestCarryAnalysis:
    def test_keys_present(self):
        from carrymath import carry_analysis

        r = carry_analysis(7, 11, 2)
        for key in (
            "N",
            "p",
            "q",
            "g",
            "h",
            "f",
            "carries",
            "convolutions",
            "carry_poly",
            "companion",
            "eigenvalues",
            "spectral_radius",
            "trace_anomaly",
        ):
            assert key in r, f"missing key: {key}"

    def test_N_correct(self):
        from carrymath import carry_analysis

        r = carry_analysis(13, 17, 2)
        assert r["N"] == 221
        assert r["trace_anomaly"] == 0


class TestBfsStates:
    def test_returns_positive_counts(self):
        from carrymath import bfs_states

        counts = bfs_states(15, 4)
        assert len(counts) == 4
        assert all(c > 0 for c in counts)

    def test_first_state_small(self):
        from carrymath import bfs_states

        counts = bfs_states(77, 4)
        assert counts[0] > 0


class TestBuildCharacterTable:
    def test_basic(self):
        from carrymath import build_character_table

        g, log_table, phi = build_character_table(7)
        assert g == 3
        assert phi == 6
        assert log_table[1] == 0
        assert len(log_table) == 6

    def test_covers_all_residues(self):
        from carrymath import build_character_table

        g, log_table, phi = build_character_table(11)
        assert set(log_table.keys()) == set(range(1, 11))


class TestRandomPrimeRange:
    @pytest.mark.parametrize("lo,hi", [(100, 200), (1000, 2000), (50, 100)])
    def test_in_range_and_prime(self, lo, hi):
        from carrymath import is_prime, random_prime_range

        p = random_prime_range(lo, hi)
        assert lo <= p < hi
        assert is_prime(p)
