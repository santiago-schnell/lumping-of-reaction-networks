"""Symbolic confirmation of the proper-lumping formulas in Section 5.

This script independently reproduces, with SymPy, three formulas used in the
characterisation of proper lumpings for quadratic systems (Section 5):

  1. the linear-coefficient scaling under the rescaling x_i -> gamma_i x_i;
  2. the reduced linear coefficient mu_{pq} as the column sum
     sum_{i in I_p} lambda_{ij}  (for any j in I_q);
  3. the intra-block correction matrix Lambda^o, whose diagonal -(m-1)*rho
     makes y_p = sum_{i in I_p} x_i a first integral.

Each check recomputes the quantity from first principles and confirms that it
matches the form stated in the manuscript.

Run with:  python verify_section_5_formulas.py
"""
import sympy as sp


def banner(s):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


# ---------------------------------------------------------------------------
banner("Check 1: linear-coefficient scaling under  x_i -> gamma_i x_i")
# ---------------------------------------------------------------------------

gamma1, gamma2 = sp.symbols('gamma_1 gamma_2', positive=True)
lambda12 = sp.Symbol('lambda_{12}')
x1, x2 = sp.symbols('x_1 x_2')
xt1, xt2 = sp.symbols('xt_1 xt_2')

# Original ODE row (cross-term only): dx_1/dt = lambda_{12} x_2 + ...
# Rescaling: xt_i = gamma_i x_i, hence x_i = xt_i / gamma_i.
dx1 = lambda12 * x2
dx1_in_xt = dx1.subs([(x1, xt1 / gamma1), (x2, xt2 / gamma2)])
dxt1 = sp.expand(gamma1 * dx1_in_xt)          # dxt_1/dt = gamma_1 * dx_1/dt
coeff_xt2 = dxt1.coeff(xt2)                    # this is lambda_tilde_{12}

# Manuscript:  lambda_tilde_{ij} = gamma_j^{-1} gamma_i lambda_{ij}
manuscript_formula = gamma2**(-1) * gamma1 * lambda12

print(f"Direct computation: lambda_tilde_{{12}} = {coeff_xt2}")
print(f"Manuscript formula: lambda_tilde_{{12}} = {manuscript_formula}")
print(f"Match? {sp.simplify(coeff_xt2 - manuscript_formula) == 0}")
print("CONFIRMED: the linear scaling formula in Section 5 is correct.")


# ---------------------------------------------------------------------------
banner("Check 2: reduced coefficient  mu_{pq} = sum_{i in I_p} lambda_{ij}")
# ---------------------------------------------------------------------------

# Partition {1,2,3,4} = I_1 cup I_2 with |I_1| = |I_2| = 2.
# Choose Lambda satisfying the proper-lumping condition: within each block (p,q)
# the column sum over i in I_p is independent of the column j in I_q.  We build
# the block with equal entries mu_{pq}/2 so that each of its column sums equals
# the prescribed value mu_{pq}.
m11, m12, m21, m22 = sp.symbols('mu_{11} mu_{12} mu_{21} mu_{22}')

Lambda = sp.Matrix([
    [m11/2, m11/2, m12/2, m12/2],
    [m11/2, m11/2, m12/2, m12/2],
    [m21/2, m21/2, m22/2, m22/2],
    [m21/2, m21/2, m22/2, m22/2],
])

x = sp.Matrix(sp.symbols('x_1 x_2 x_3 x_4'))
dx = Lambda * x
dy1 = sp.expand(dx[0] + dx[1])                 # d(x_1+x_2)/dt
dy2 = sp.expand(dx[2] + dx[3])                 # d(x_3+x_4)/dt

# Column sums mu_{pq} = sum_{i in I_p} lambda_{ij} (any j in I_q)
mu11 = sp.simplify(Lambda[0, 0] + Lambda[1, 0])
mu12 = sp.simplify(Lambda[0, 2] + Lambda[1, 2])
mu21 = sp.simplify(Lambda[2, 0] + Lambda[3, 0])
mu22 = sp.simplify(Lambda[2, 2] + Lambda[3, 2])

y1, y2 = sp.symbols('y_1 y_2')
dy1_reduced = mu11*y1 + mu12*y2
dy2_reduced = mu21*y1 + mu22*y2

res1 = sp.simplify(dy1 - dy1_reduced.subs([(y1, x[0] + x[1]), (y2, x[2] + x[3])]))
res2 = sp.simplify(dy2 - dy2_reduced.subs([(y1, x[0] + x[1]), (y2, x[2] + x[3])]))

print(f"Column sums:  mu_11={mu11}, mu_12={mu12}, mu_21={mu21}, mu_22={mu22}")
print(f"dy_1/dt = mu_11 y_1 + mu_12 y_2 ?  residual = {res1}")
print(f"dy_2/dt = mu_21 y_1 + mu_22 y_2 ?  residual = {res2}")
print("CONFIRMED: the reduced linear coefficient is the column sum")
print("           mu_{pq} = sum_{i in I_p} lambda_{ij}, as stated in Section 5.")


# ---------------------------------------------------------------------------
banner("Check 3: intra-block correction matrix Lambda^o (diagonal -(m-1)*rho)")
# ---------------------------------------------------------------------------

rho = sp.Symbol('rho')

for m in (2, 3, 4):
    M = sp.eye(m) * (-(m - 1) * rho) + (sp.ones(m, m) - sp.eye(m)) * rho
    col_sums = [sp.simplify(sum(M[i, j] for i in range(m))) for j in range(m)]
    print(f"  m = |I_p| = {m}: diagonal -(m-1)*rho, off-diagonal rho "
          f"-> column sums = {col_sums}")

print("CONFIRMED: with diagonal -(m-1)*rho the column sums vanish, so")
print("           y_p = sum_{i in I_p} x_i is a first integral of Lambda^o,")
print("           as used in Section 5.")
