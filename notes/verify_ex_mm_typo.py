"""Symbolic verification of the typo in Example ex:mm.

This script verifies the claim made in the review: the labels y_1 and y_3 are
swapped in the printed dynamics at the end of Example ex:mm (paper line 1052-1054).

Run with:  python verify_ex_mm_typo.py
"""
import sympy as sp

# State variables and parameters
s, e, c, p = sp.symbols('s e c p', positive=True)
k1, km1, k2, km2 = sp.symbols('k_1 k_{-1} k_2 k_{-2}', positive=True)

# Reversible Michaelis-Menten ODEs (paper line 964-968)
ds = -k1*e*s + km1*c
de = -k1*e*s + (km1 + k2)*c - km2*e*p
dc =  k1*e*s - (km1 + k2)*c + km2*e*p
dp =  k2*c - km2*e*p

# Print the two stoichiometric first integrals (true for all parameter values)
print("=== Stoichiometric first integrals (true for all parameters) ===")
print(f"  d(s+c+p)/dt = {sp.simplify(ds + dc + dp)}")  # should be 0
print(f"  d(e+c)/dt   = {sp.simplify(de + dc)}")       # should be 0

# Note: s+p is NOT a first integral in general
print(f"\n  d(s+p)/dt   = {sp.simplify(ds + dp)}     (NOT zero)")

# Now specialize to the critical-parameter value k_1 = k_{-2}
print("\n=== Specialize to critical parameter  k_1 = k_{-2} ===")
ds_c = ds.subs(km2, k1)
de_c = de.subs(km2, k1)
dc_c = dc.subs(km2, k1)
dp_c = dp.subs(km2, k1)

print(f"  d(s+p)/dt at k_1=k_{{-2}}: {sp.factor(ds_c + dp_c)}")

# The text on line 1048-1049 defines:
#    y_1 = s+p,  y_2 = e+c,  y_3 = s+p+c
print("\n=== Paper's labels:  y_1 = s+p,  y_2 = e+c,  y_3 = s+p+c ===")

dy1 = sp.simplify(ds_c + dp_c)
dy2 = sp.simplify(de_c + dc_c)
dy3 = sp.simplify(ds_c + dp_c + dc_c)

print(f"  dy_1/dt (s+p)         = {dy1}")
print(f"  dy_2/dt (e+c)         = {dy2}")
print(f"  dy_3/dt (s+p+c)       = {dy3}")

# Express dy_1/dt in terms of y_1, y_2, y_3.  With y_1=s+p, y_2=e+c, y_3=s+p+c:
#   c = y_3 - y_1,   e = y_2 - c = y_1 + y_2 - y_3.
y1, y2, y3 = sp.symbols('y_1 y_2 y_3')
dy1_pred = -k1*(y1 + y2 - y3)*y1 + (km1 + k2)*(y3 - y1)

# Verify by direct substitution
verify = dy1_pred.subs([(y1, s+p), (y2, e+c), (y3, s+p+c)])
diff = sp.simplify(dy1 - verify)
print(f"\nProposed expression for dy_1/dt:")
print(f"  dy_1/dt = {dy1_pred}")
print(f"Verification (should print 0): {diff}")

print("\n========================================================")
print("CONCLUSION")
print("========================================================")
print("With the paper's labels y_1=s+p, y_2=e+c, y_3=s+p+c, the correct dynamics")
print("at k_1 = k_{-2} are:")
print()
print("    dy_1/dt = -k_1 (y_1+y_2-y_3) y_1 + (k_{-1}+k_2)(y_3-y_1)")
print("    dy_2/dt = 0")
print("    dy_3/dt = 0")
print()
print("But paper line 1052-1054 prints these with y_1 and y_3 SWAPPED:")
print()
print("    dy_1 = 0                                       <-- WRONG (should be dy_3)")
print("    dy_2 = 0")
print("    dy_3 = -k_1 (y_1+y_2-y_3) y_1 + (k_{-1}+k_2)(y_3-y_1)  <-- WRONG (should be dy_1)")
print()
print("Conclusion: the labels y_1 and y_3 must be exchanged in the printed system.")
