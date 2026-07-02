#!/usr/bin/env python3
"""
Figure: Type 1 vs Type 2 invariant subspaces for a single reaction.
Illustrates the geometric meaning of the two types from Proposition 3.1.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "generated"
OUT_DIR.mkdir(exist_ok=True)
from matplotlib.patches import FancyArrowPatch, Polygon
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

# Set up figure style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# ============================================================
# Panel (a): Type 1 - Non-reactant species
# ============================================================
ax1 = axes[0]

# Draw coordinate axes
ax1.annotate('', xy=(3.2, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1))
ax1.annotate('', xy=(0, 3.2), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1))
ax1.annotate('', xy=(2.2, 2.2), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

ax1.text(3.3, 0, r'$x_1$', fontsize=11, ha='left', va='center')
ax1.text(0, 3.4, r'$x_2$', fontsize=11, ha='center', va='bottom')
ax1.text(2.35, 2.35, r'$x_3$', fontsize=11, ha='left', va='bottom')

# Type 1 subspace: span of non-reactant species (e.g., x3 axis)
# Draw the subspace as a line
t = np.linspace(0, 2.5, 50)
ax1.plot(t * 0.707, t * 0.707, 'b-', linewidth=2.5, label='Type 1 subspace')

# Shade the subspace region slightly
ax1.fill([0, 1.8, 1.8, 0], [0, 1.8, 2.0, 0.2], color='blue', alpha=0.1)

# Mark the reaction vector v (should NOT be in this subspace for Type 1)
v = np.array([1.5, 2.0])  # reaction vector in (x1, x2) plane
ax1.annotate('', xy=v, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax1.text(v[0] + 0.15, v[1] + 0.1, r'$v$', fontsize=12, color='red', fontweight='bold')

# Label
ax1.text(1.5, 0.6, r'$W = \mathrm{span}\{e_3\}$', fontsize=10, color='blue',
         rotation=45, ha='center', va='bottom')

ax1.set_xlim(-0.3, 3.5)
ax1.set_ylim(-0.3, 3.5)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title(r'(a) Type 1: Non-reactant species', fontsize=11, pad=10)

# Add explanation text
ax1.text(1.75, -0.6, r'$X_3$ not a reactant $\Rightarrow e_3 \in W$', 
         fontsize=9, ha='center', style='italic')
ax1.text(1.75, -1.0, r'$D\varphi(x) \cdot W = \{0\}$ for all $x$', 
         fontsize=9, ha='center', style='italic')

# ============================================================
# Panel (b): Type 2 - Stoichiometric first integral
# ============================================================
ax2 = axes[1]

# Draw coordinate axes
ax2.annotate('', xy=(3.2, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1))
ax2.annotate('', xy=(0, 3.2), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1))
ax2.annotate('', xy=(2.2, 2.2), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

ax2.text(3.3, 0, r'$x_1$', fontsize=11, ha='left', va='center')
ax2.text(0, 3.4, r'$x_2$', fontsize=11, ha='center', va='bottom')
ax2.text(2.35, 2.35, r'$x_3$', fontsize=11, ha='left', va='bottom')

# Type 2 subspace: contains the reaction vector v
v = np.array([1.2, 1.8])  # reaction vector
ax2.annotate('', xy=v, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax2.text(v[0] - 0.25, v[1] + 0.1, r'$v$', fontsize=12, color='red', fontweight='bold')

# Draw the subspace containing v (a plane through origin containing v)
# We'll show it as a shaded region
t = np.linspace(-0.5, 2.5, 50)
# Subspace spanned by v and some other vector
ax2.plot(t * v[0] / np.linalg.norm(v) * 1.5, t * v[1] / np.linalg.norm(v) * 1.5, 
         'b-', linewidth=2.5, label='Type 2 subspace')

# Shade the subspace
verts = [(0, 0), (v[0]*1.3, v[1]*1.3), (v[0]*1.3 + 0.8, v[1]*1.3 - 0.5), (0.8, -0.5)]
poly = Polygon(verts, closed=True, facecolor='blue', alpha=0.15, edgecolor='none')
ax2.add_patch(poly)

# Add another basis vector for the subspace
w = np.array([1.8, 0.3])
ax2.annotate('', xy=w, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, linestyle='--'))

ax2.text(2.0, 1.2, r'$W \ni v$', fontsize=10, color='blue', ha='center')

ax2.set_xlim(-0.3, 3.5)
ax2.set_ylim(-0.8, 3.5)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title(r'(b) Type 2: Stoichiometric first integral', fontsize=11, pad=10)

# Add explanation text
ax2.text(1.75, -1.0, r'$v \in W \Rightarrow$ rows of $T$ are first integrals', 
         fontsize=9, ha='center', style='italic')
ax2.text(1.75, -1.4, r'$T \cdot v = 0$', 
         fontsize=9, ha='center', style='italic')

plt.tight_layout()
plt.savefig(OUT_DIR / 'type1_type2_subspaces.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(OUT_DIR / 'type1_type2_subspaces.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Type 1 vs Type 2 subspaces figure created!")
