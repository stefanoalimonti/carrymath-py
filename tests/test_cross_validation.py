"""Cross-validation tests with hardcoded expected values.

These EXACT same test vectors are replicated in carrymath-js/tests/cross_validation.test.js.
Any divergence between the two libraries will be caught here.
"""

import pytest

from carrymath import CarryChain, carry_chain, carry_polynomial, convolution, from_digits, to_digits
from carrymath.digits import eval_poly

VECTORS = [
    {
        "p": 3,
        "q": 5,
        "N": 15,
        "g_digits": [1, 1],
        "h_digits": [1, 0, 1],
        "carries": [0, 0, 0, 0, 0],
        "convolution": [1, 1, 1, 1],
        "carry_poly": [0],
        "trace_anomaly": 0,
    },
    {
        "p": 7,
        "q": 11,
        "N": 77,
        "g_digits": [1, 1, 1],
        "h_digits": [1, 1, 0, 1],
        "carries": [0, 0, 1, 1, 1, 1, 1],
        "convolution": [1, 2, 2, 2, 1, 1],
        "carry_poly": [0, -1, -1, -1, -1, -1],
        "trace_anomaly": 1,
    },
    {
        "p": 13,
        "q": 17,
        "N": 221,
        "g_digits": [1, 0, 1, 1],
        "h_digits": [1, 0, 0, 0, 1],
        "carries": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "convolution": [1, 0, 1, 1, 1, 0, 1, 1],
        "carry_poly": [0],
        "trace_anomaly": 0,
    },
    {
        "p": 127,
        "q": 131,
        "N": 16637,
        "g_digits": [1, 1, 1, 1, 1, 1, 1],
        "h_digits": [1, 1, 0, 0, 0, 0, 0, 1],
        "carries": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "trace_anomaly": 1,
    },
    {
        "p": 1009,
        "q": 1013,
        "N": 1022117,
        "carries_prefix": [0, 0, 0, 0, 0, 1, 1, 2, 2, 3],
        "trace_anomaly": 2,
    },
    {
        "p": 65537,
        "q": 65539,
        "N": 4295229443,
        "g_digits": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "h_digits": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "carries_prefix": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "trace_anomaly": 0,
    },
    {
        "p": 4294967311,
        "q": 4294967357,
        "N": 18446744400127067027,
        "carries_prefix": [0, 0, 0, 1, 2, 2, 3, 3, 2, 1],
        "carry_poly_prefix": [0, 0, -1, -2, -2, -3, -3, -2, -1, 0],
        "trace_anomaly": 0,
    },
]


def _vector_id(v):
    return f"{v['p']}*{v['q']}"


@pytest.mark.parametrize("v", VECTORS, ids=[_vector_id(v) for v in VECTORS])
class TestCrossValidation:
    def test_to_digits(self, v):
        p, q = v["p"], v["q"]
        gd = to_digits(p, 2)
        hd = to_digits(q, 2)
        if "g_digits" in v:
            assert gd == v["g_digits"], f"to_digits({p})"
            assert hd == v["h_digits"], f"to_digits({q})"

    def test_from_digits_roundtrip(self, v):
        p, q = v["p"], v["q"]
        assert from_digits(to_digits(p, 2), 2) == p
        assert from_digits(to_digits(q, 2), 2) == q

    def test_carry_chain(self, v):
        gd = to_digits(v["p"], 2)
        hd = to_digits(v["q"], 2)
        carries = carry_chain(gd, hd)
        if "carries" in v:
            assert carries == v["carries"]
        if "carries_prefix" in v:
            prefix = v["carries_prefix"]
            assert carries[: len(prefix)] == prefix

    def test_convolution(self, v):
        if "convolution" not in v:
            pytest.skip("no convolution vector")
        gd = to_digits(v["p"], 2)
        hd = to_digits(v["q"], 2)
        assert convolution(gd, hd) == v["convolution"]

    def test_carry_polynomial(self, v):
        cpoly = carry_polynomial(v["p"], v["q"])
        if "carry_poly" in v:
            assert cpoly == v["carry_poly"]
        if "carry_poly_prefix" in v:
            prefix = v["carry_poly_prefix"]
            assert cpoly[: len(prefix)] == prefix

    def test_carry_representation_theorem(self, v):
        p, q, N = v["p"], v["q"], v["N"]
        gd = to_digits(p, 2)
        hd = to_digits(q, 2)
        fd = to_digits(N, 2)
        cpoly = carry_polynomial(p, q)
        for x in [3, 10, 100]:
            lhs = eval_poly(gd, x) * eval_poly(hd, x)
            rhs = eval_poly(fd, x) + (x - 2) * eval_poly(cpoly, x)
            assert lhs == rhs, f"CRT failed at x={x}"

    def test_product_digits(self, v):
        p, q, N = v["p"], v["q"], v["N"]
        cc = CarryChain.from_integers(p, q)
        assert from_digits(cc.product_digits, 2) == N

    def test_trace_anomaly(self, v):
        if "trace_anomaly" not in v:
            pytest.skip("no trace_anomaly vector")
        cc = CarryChain.from_integers(v["p"], v["q"])
        assert cc.trace_anomaly() == v["trace_anomaly"]
