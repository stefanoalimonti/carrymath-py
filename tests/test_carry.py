"""Tests for core carry chain functionality."""

from fractions import Fraction

from carrymath import CarryChain, carry_chain, carry_polynomial, convolution, from_digits, to_digits
from carrymath.covariance import bulk_covariance, verify_lemma_d
from carrymath.enumerate import enumerate_exact


def test_to_from_digits():
    for n in [0, 1, 13, 143, 255, 1024]:
        assert from_digits(to_digits(n, 2), 2) == n
        assert from_digits(to_digits(n, 10), 10) == n


def test_carry_chain_simple():
    # 3 * 3 = 9: g = [1, 1], h = [1, 1]
    carries = carry_chain([1, 1], [1, 1])
    # conv_0 = 1, total=1, carry=0
    # conv_1 = 2, total=2, carry=1
    # conv_2 = 1, total=2, carry=1
    assert carries[0] == 0
    assert carries[1] == 0
    assert carries[2] == 1


def test_carry_chain_from_integers():
    cc = CarryChain.from_integers(7, 5)
    assert cc.carries[0] == 0  # always
    product = from_digits(cc.product_digits, 2)
    assert product == 35


def test_convolution():
    assert convolution([1], [1]) == [1]
    assert convolution([1, 1], [1, 1]) == [1, 2, 1]
    assert convolution([1, 0, 1], [1, 1]) == [1, 1, 1, 1]


def test_carry_polynomial():
    # P(x)Q(x) = F(x) + (x-2)C(x)
    for p, q in [(3, 5), (7, 11), (13, 17)]:
        cp = carry_polynomial(p, q)
        gd = to_digits(p)
        hd = to_digits(q)
        fd = to_digits(p * q)
        for x in [3, 5, 10]:
            from carrymath.digits import eval_poly

            lhs = eval_poly(gd, x) * eval_poly(hd, x)
            rhs = eval_poly(fd, x) + (x - 2) * eval_poly(cp, x)
            assert lhs == rhs, f"CRT failed for {p}*{q} at x={x}"


def test_enumerate_exact():
    r = enumerate_exact(3)
    # E[carry_j] = (j-1)/4 for j >= 1
    for j in range(1, r["D"] + 1):
        if j < len(r["E_carry"]):
            expected = Fraction(j - 1, 4)
            assert r["E_carry"][j] == expected, f"E[carry_{j}] = {r['E_carry'][j]} != {expected}"


def test_bulk_covariance_odd():
    r = bulk_covariance(6)
    for j in range(3, r["D"], 2):
        expected = Fraction(j - 1, 8)
        assert r["cov"][j] == expected, f"Cov at j={j}: {r['cov'][j]} != {expected}"


def test_lemma_d():
    result = verify_lemma_d(4)
    assert result["lemma_d_holds"], "Lemma D failed at K=4"


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"  PASS: {name}")
            except AssertionError as e:
                print(f"  FAIL: {name}: {e}")
    print("All tests complete.")
