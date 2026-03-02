"""Verify the Universal Off-Diagonal Covariance (Lemma D).

Proves: Cov(carry_j, g_i h_{j-i}) = 1/8 for all off-diagonal pairs i != j-i.
"""

from carrymath.covariance import bulk_covariance, even_j_corrections, verify_lemma_d

# Verify Lemma D at K=4 (32 configurations)
print("=" * 60)
print("Lemma D Verification (K=4)")
print("=" * 60)
result = verify_lemma_d(4)
print(f"Lemma D holds: {result['lemma_d_holds']}")
print(f"Off-diagonal pairs tested: {result['n_interior_offdiag']}")
print(f"Off-diagonal pairs matching 1/8: {result['n_interior_offdiag_match']}")
print()

# Bulk covariance formula check
print("=" * 60)
print("Cov(carry_j, conv_j) = (j-1)/8 for odd j (K=6)")
print("=" * 60)
r = bulk_covariance(6)
for j in sorted(r['formula_check'].keys()):
    fc = r['formula_check'][j]
    status = "MATCH" if fc['match'] else "MISMATCH"
    print(f"  j={j}: computed={fc['computed']}, formula={(j-1)}/8={fc['formula']}  [{status}]")
print()

# Even-j corrections
print("=" * 60)
print("Even-j corrections epsilon_j (K=8)")
print("=" * 60)
eps = even_j_corrections(8)
for j in sorted(eps.keys()):
    print(f"  j={j}: epsilon = {eps[j]} = {float(eps[j]):.6f}")
