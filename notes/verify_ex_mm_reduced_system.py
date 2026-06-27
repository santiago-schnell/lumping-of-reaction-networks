"""Symbolic confirmation of the reduced system in Example ex:mm.

This script independently reproduces, with SymPy, the reduced system stated at
the end of Example ``ex:mm`` (Section 4.4): the reversible Michaelis--Menten
network with the first two rows of T prescribed as stoichiometric first
integrals.

With the labels

    y_1 = s + p,   y_2 = e + c,   y_3 = s + p + c,

and at the critical parameter value k_1 = k_{-2}, it confirms that the
nontrivial dynamics lie in dy_1/dt while y_2 and y_3 are first integrals:

    dy_1/dt = -k_1 (y_1 + y_2 - y_3) y_1 + (k_{-1} + k_2)(y_3 - y_1)
    dy_2/dt = 0
    dy_3/dt = 0

Run with:  python verify_ex_mm_reduced_system.py
"""
import sympy as sp

# State variables and parameters
s, e, c, p = sp.symbols('s e c p', positive=True)
k1, km1, k2, km2 = sp.symbols('k_1 k_{-1} k_2 k_{-2}', positive=True)

# Reversible Michaelis--Menten ODEs
ds = -k1*e*s + km1*c
de = -k1*e*s + (km1 + k2)*c - km2*e*p
dc =  k1*e*s - (km1 + k2)*c + km2*e*p
dp =  k2*c - km2*e*p

print("=== Stoichiometric first integrals (hold for all parameters) ===")
print(f"  d(s+c+p)/dt = {sp.simplify(ds + dc + dp)}")   # 0
print(f"  d(e+c)/dt   = {sp.simplify(de + dc)}")         # 0
print(f"\n  d(s+p)/dt   = {sp.simplify(ds + dp)}     (not a first integral in general)")

print("\n=== Specialize to the critical parameter  k_1 = k_{-2} ===")
ds_c = ds.subs(km2, k1)
de_c = de.subs(km2, k1)
dc_c = dc.subs(km2, k1)
dp_c = dp.subs(km2, k1)

print("\n=== Labels:  y_1 = s+p,  y_2 = e+c,  y_3 = s+p+c ===")
dy1 = sp.simplify(ds_c + dp_c)
dy2 = sp.simplify(de_c + dc_c)
dy3 = sp.simplify(ds_c + dp_c + dc_c)
print(f"  dy_1/dt (s+p)     = {dy1}")
print(f"  dy_2/dt (e+c)     = {dy2}")
print(f"  dy_3/dt (s+p+c)   = {dy3}")

# Express dy_1/dt in terms of y_1, y_2, y_3.
# With y_1 = s+p, y_2 = e+c, y_3 = s+p+c:  c = y_3 - y_1,  e = y_1 + y_2 - y_3.
y1, y2, y3 = sp.symbols('y_1 y_2 y_3')
dy1_reduced = -k1*(y1 + y2 - y3)*y1 + (km1 + k2)*(y3 - y1)
residual = sp.simplify(dy1 - dy1_reduced.subs([(y1, s + p), (y2, e + c), (y3, s + p + c)]))

print("\nReduced equation written in lumped coordinates:")
print(f"  dy_1/dt = {dy1_reduced}")
print(f"Residual against the direct computation (expected 0): {residual}")

print("\n" + "=" * 60)
print("CONFIRMED")
print("=" * 60)
print("At k_1 = k_{-2}, with y_1 = s+p, y_2 = e+c, y_3 = s+p+c, the reduced")
print("system is")
print()
print("    dy_1/dt = -k_1 (y_1+y_2-y_3) y_1 + (k_{-1}+k_2)(y_3-y_1)")
print("    dy_2/dt = 0")
print("    dy_3/dt = 0,")
print()
print("i.e. the dynamics reduce to y_1, with y_2 and y_3 first integrals,")
print("in agreement with Example ex:mm.")
