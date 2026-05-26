"""Symbolic verification of the three algebraic errors in Section 5
of the manuscript that ChatGPT identified.

Run with:  python verify_section_5_errors.py

The three errors:
  1. The linear-coefficient scaling formula has gamma_i and gamma_j inverted.
  2. The reduced coefficient mu_{pq} is off by a factor of |I_p|.
  3. The dissipative correction matrix has diagonal -m*rho where -(m-1)*rho
     is needed to make column sums vanish.
"""
import sympy as sp


def banner(s):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


# ---------------------------------------------------------------------------
banner("Error 1: Linear-coefficient scaling formula (paper line 1161)")
# ---------------------------------------------------------------------------

gamma1, gamma2 = sp.symbols('gamma_1 gamma_2', positive=True)
lambda12 = sp.Symbol('lambda_{12}')
x1, x2 = sp.symbols('x_1 x_2')
xt1, xt2 = sp.symbols('xt_1 xt_2')

# Original ODE row: dx_1/dt = lambda_{11} x_1 + lambda_{12} x_2 + (quadratic terms)
# Scaling: x_tilde_i = gamma_i * x_i, so x_i = x_tilde_i / gamma_i.
dx1 = lambda12 * x2                       # focus on the cross-term coefficient
dx1_in_xt = dx1.subs([(x1, xt1/gamma1), (x2, xt2/gamma2)])
dxt1 = sp.expand(gamma1 * dx1_in_xt)      # dxt_1/dt = gamma_1 * dx_1/dt
coeff_xt2 = dxt1.coeff(xt2)               # this is lambda_tilde_{12}

paper_formula   = gamma1**(-1) * gamma2 * lambda12     # paper line 1161
correct_formula = gamma1 * gamma2**(-1) * lambda12

print(f"Direct computation:    lambda_tilde_{{12}} = {coeff_xt2}")
print(f"Paper line 1161:       lambda_tilde_{{12}} = {paper_formula}")
print(f"Correct formula:       lambda_tilde_{{12}} = {correct_formula}")
print()
print(f"Direct == Paper?    {sp.simplify(coeff_xt2 - paper_formula) == 0}")
print(f"Direct == Correct?  {sp.simplify(coeff_xt2 - correct_formula) == 0}")
print()
print("VERDICT: The paper formula has the gamma's inverted.")


# ---------------------------------------------------------------------------
banner("Error 2: mu_{pq} reduced coefficient (paper line 1328-1333)")
# ---------------------------------------------------------------------------

# Set up a partition {1,2,3,4} = I_1 cup I_2 with |I_1| = |I_2| = 2.
# Use a Lambda that satisfies the proper-lumping column-sum condition:
# all entries within each block (p,q) equal so column sums are well-defined.

c11, c12, c21, c22 = sp.symbols('c_{11} c_{12} c_{21} c_{22}')

# Construct Lambda so that the column sum on block (p,q) is exactly c_{pq}.
# With |I_p| = 2 we set each entry = c_{pq} / 2.
Lambda = sp.Matrix([
    [c11/2, c11/2, c12/2, c12/2],
    [c11/2, c11/2, c12/2, c12/2],
    [c21/2, c21/2, c22/2, c22/2],
    [c21/2, c21/2, c22/2, c22/2],
])

x = sp.Matrix(sp.symbols('x_1 x_2 x_3 x_4'))
dx = Lambda * x
dy1 = sp.simplify(dx[0] + dx[1])          # d(x_1+x_2)/dt
dy2 = sp.simplify(dx[2] + dx[3])

# Express in terms of y_1 = x_1+x_2, y_2 = x_3+x_4
y1, y2 = sp.symbols('y_1 y_2')
print(f"dy_1/dt (direct) = {dy1}")
print(f"  Coefficient of (x_1+x_2) = {dy1.coeff(x[0])}   (must equal coeff of x_2 too)")
print(f"  Coefficient of (x_3+x_4) = {dy1.coeff(x[2])}")
print()
print("So the reduced equation is")
print("  dy_1/dt = c_{11} * y_1 + c_{12} * y_2     [column sums!]")
print("NOT     dy_1/dt = mu_{11} * y_1 + ...       [where mu = c / |I_p|]")
print()
print("VERDICT: The paper's mu_{pq} = c_{pq}/|I_p| is an average, but the")
print("         reduced equation requires the column sum c_{pq} itself.")
print("         The paper's formula is off by a factor of |I_p|.")


# ---------------------------------------------------------------------------
banner("Error 3: Dissipative correction matrix diagonal (paper line 1353-1359)")
# ---------------------------------------------------------------------------

rho = sp.Symbol('rho')

def block(diag_factor, m):
    """Build the m x m intra-block correction matrix with off-diagonals = rho
    and diagonal = diag_factor."""
    M = sp.eye(m) * diag_factor + (sp.ones(m, m) - sp.eye(m)) * rho
    return M

for m in (2, 3, 4):
    print(f"--- m = |I_p| = {m} ---")
    M_paper = block(-m * rho, m)
    M_correct = block(-(m - 1) * rho, m)
    col_sums_paper   = [sum(M_paper[i, j]   for i in range(m)) for j in range(m)]
    col_sums_correct = [sum(M_correct[i, j] for i in range(m)) for j in range(m)]
    print(f"  Paper   (diag = -m*rho = -{m}*rho): column sums = {col_sums_paper}")
    print(f"  Correct (diag = -(m-1)*rho)       : column sums = {col_sums_correct}")
    print()

print("VERDICT: Paper's matrix has column sums = -rho (NOT zero), so y_p is")
print("         NOT a first integral of Lambda^o.  Diagonal must be")
print("         -(m-1)*rho so that column sums vanish.")
