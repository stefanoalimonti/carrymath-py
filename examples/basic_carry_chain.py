"""Basic usage of the carrymath library: carry chains, polynomials, expectations."""

from carrymath import CarryChain, carry_chain, carry_polynomial, to_digits
from carrymath.primes import random_semiprime

# 1. Compute the carry chain for 13 * 11 = 143
p, q = 13, 11
g = to_digits(p, 2)  # [1, 0, 1, 1]
h = to_digits(q, 2)  # [1, 1, 0, 1]
carries = carry_chain(g, h)
print(f"{p} * {q} = {p*q}")
print(f"g (base-2 digits of {p}): {g}")
print(f"h (base-2 digits of {q}): {h}")
print(f"Carry chain: {carries}")
print()

# 2. Using the CarryChain object
cc = CarryChain.from_integers(p, q)
print(f"CarryChain: {cc}")
print(f"Carries:     {cc.carries}")
print(f"Convolutions: {cc.convolutions}")
print(f"Trace anomaly (c_{{D-1}}): {cc.trace_anomaly()}")
print()

# 3. Carry polynomial
cpoly = carry_polynomial(p, q)
print(f"Carry polynomial C(x) coefficients: {cpoly}")
print()

# 4. Random semiprime
N, p, q = random_semiprime(20)
print(f"Random 20-bit semiprime: N = {N} = {p} * {q}")
cc = CarryChain.from_integers(p, q)
print(f"Carry chain length: {cc.D}")
print(f"Carries: {cc.carries}")
