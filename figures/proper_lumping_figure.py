#!/usr/bin/env python3
"""
Figure: Block structure for proper lumping (Proposition 5.1).
Illustrates the equal column sum condition on Jacobian blocks.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "generated"
OUT_DIR.mkdir(exist_ok=True)
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

# Set up figure style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# ============================================================
# Panel (a): Species partition and lumping matrix T
# ============================================================
ax1 = axes[0]

# Draw the species partition
block_colors = ['#3498db', '#e74c3c', '#2ecc71']
block_labels = [r'$I_1 = \{1,2\}$', r'$I_2 = \{3,4\}$', r'$I_3 = \{5\}$']

# Species boxes
y_pos = 2.5
box_width = 0.6
box_height = 0.5
spacing = 0.15

species = [r'$X_1$', r'$X_2$', r'$X_3$', r'$X_4$', r'$X_5$']
block_assignments = [0, 0, 1, 1, 2]  # which block each species belongs to

x_positions = []
x = 0.5
for i, (sp, block) in enumerate(zip(species, block_assignments)):
    rect = FancyBboxPatch((x, y_pos), box_width, box_height,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=block_colors[block], edgecolor='black',
                          alpha=0.7, linewidth=1)
    ax1.add_patch(rect)
    ax1.text(x + box_width/2, y_pos + box_height/2, sp, 
             fontsize=11, ha='center', va='center', fontweight='bold')
    x_positions.append(x + box_width/2)
    x += box_width + spacing

# Draw block brackets
bracket_y = y_pos + box_height + 0.15
ax1.plot([x_positions[0] - 0.25, x_positions[0] - 0.25, x_positions[1] + 0.25, x_positions[1] + 0.25],
         [bracket_y, bracket_y + 0.15, bracket_y + 0.15, bracket_y], 'k-', lw=1)
ax1.text((x_positions[0] + x_positions[1])/2, bracket_y + 0.25, block_labels[0], 
         fontsize=9, ha='center', va='bottom')

ax1.plot([x_positions[2] - 0.25, x_positions[2] - 0.25, x_positions[3] + 0.25, x_positions[3] + 0.25],
         [bracket_y, bracket_y + 0.15, bracket_y + 0.15, bracket_y], 'k-', lw=1)
ax1.text((x_positions[2] + x_positions[3])/2, bracket_y + 0.25, block_labels[1], 
         fontsize=9, ha='center', va='bottom')

ax1.plot([x_positions[4] - 0.25, x_positions[4] - 0.25, x_positions[4] + 0.25, x_positions[4] + 0.25],
         [bracket_y, bracket_y + 0.15, bracket_y + 0.15, bracket_y], 'k-', lw=1)
ax1.text(x_positions[4], bracket_y + 0.25, block_labels[2], 
         fontsize=9, ha='center', va='bottom')

# Draw arrow to lumped variables
ax1.annotate('', xy=(2.2, 1.3), xytext=(2.2, 2.4),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax1.text(2.5, 1.85, 'Proper\nlumping', fontsize=9, ha='left', va='center')

# Lumped variables
y_lump = 0.7
lump_vars = [r'$y_1 = x_1 + x_2$', r'$y_2 = x_3 + x_4$', r'$y_3 = x_5$']
lump_x = [0.8, 2.2, 3.6]
for i, (lv, lx) in enumerate(zip(lump_vars, lump_x)):
    rect = FancyBboxPatch((lx - 0.5, y_lump), 1.0, 0.45,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=block_colors[i], edgecolor='black',
                          alpha=0.4, linewidth=1)
    ax1.add_patch(rect)
    ax1.text(lx, y_lump + 0.22, lv, fontsize=10, ha='center', va='center')

ax1.set_xlim(0, 4.5)
ax1.set_ylim(0.2, 3.6)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title(r'(a) Species partition $\to$ lumped variables', fontsize=11, pad=10)

# ============================================================
# Panel (b): Jacobian block structure with equal column sums
# ============================================================
ax2 = axes[1]

# Draw the block matrix
matrix_left = 0.3
matrix_bottom = 0.5
cell_size = 0.5
n_rows, n_cols = 5, 5

# Block structure: 2x2, 2x2, 1x1 on diagonal; off-diagonal blocks
block_structure = [
    (0, 0, 2, 2),  # M_11
    (0, 2, 2, 2),  # M_12
    (0, 4, 2, 1),  # M_13
    (2, 0, 2, 2),  # M_21
    (2, 2, 2, 2),  # M_22
    (2, 4, 2, 1),  # M_23
    (4, 0, 1, 2),  # M_31
    (4, 2, 1, 2),  # M_32
    (4, 4, 1, 1),  # M_33
]

block_labels_mat = [
    r'$M_{11}$', r'$M_{12}$', r'$M_{13}$',
    r'$M_{21}$', r'$M_{22}$', r'$M_{23}$',
    r'$M_{31}$', r'$M_{32}$', r'$M_{33}$'
]

# Draw grid
for i in range(n_rows + 1):
    lw = 1.5 if i in [0, 2, 4, 5] else 0.5
    ax2.plot([matrix_left, matrix_left + n_cols * cell_size],
             [matrix_bottom + i * cell_size, matrix_bottom + i * cell_size],
             'k-', lw=lw)
for j in range(n_cols + 1):
    lw = 1.5 if j in [0, 2, 4, 5] else 0.5
    ax2.plot([matrix_left + j * cell_size, matrix_left + j * cell_size],
             [matrix_bottom, matrix_bottom + n_rows * cell_size],
             'k-', lw=lw)

# Shade blocks and add labels
for idx, (row, col, h, w) in enumerate(block_structure):
    x = matrix_left + col * cell_size
    y = matrix_bottom + (n_rows - row - h) * cell_size
    
    # Shade diagonal blocks differently
    if row // 2 == col // 2 or (row == 4 and col == 4):
        color = '#ecf0f1'
    else:
        color = '#fdf2e9'
    
    rect = Rectangle((x, y), w * cell_size, h * cell_size,
                      facecolor=color, edgecolor='none', alpha=0.7)
    ax2.add_patch(rect)
    
    # Add block label
    ax2.text(x + w * cell_size / 2, y + h * cell_size / 2,
             block_labels_mat[idx], fontsize=9, ha='center', va='center')

# Add row/column labels
for i in range(n_rows):
    ax2.text(matrix_left - 0.15, matrix_bottom + (n_rows - i - 0.5) * cell_size,
             str(i + 1), fontsize=9, ha='right', va='center')
for j in range(n_cols):
    ax2.text(matrix_left + (j + 0.5) * cell_size, matrix_bottom + n_rows * cell_size + 0.1,
             str(j + 1), fontsize=9, ha='center', va='bottom')

# Matrix label
ax2.text(matrix_left - 0.4, matrix_bottom + n_rows * cell_size / 2,
         r'$DF(x)$', fontsize=12, ha='right', va='center', rotation=90)

# Add the equal column sum condition
ax2.text(matrix_left + n_cols * cell_size + 0.3, matrix_bottom + n_rows * cell_size - 0.3,
         'Column sums\nequal within\neach block', fontsize=9, ha='left', va='top',
         style='italic')

# Draw arrows indicating column sums
arrow_y = matrix_bottom - 0.15
ax2.annotate('', xy=(matrix_left + 0.25, arrow_y), 
             xytext=(matrix_left + 0.75, arrow_y),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
ax2.text(matrix_left + 0.5, arrow_y - 0.12, r'$=$', fontsize=10, 
         ha='center', va='top', color='black')

ax2.annotate('', xy=(matrix_left + 1.0 + 0.25, arrow_y), 
             xytext=(matrix_left + 1.0 + 0.75, arrow_y),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
ax2.text(matrix_left + 1.5, arrow_y - 0.12, r'$=$', fontsize=10, 
         ha='center', va='top', color='black')

# Condition text at bottom - moved lower to avoid overlap
ax2.text(matrix_left + n_cols * cell_size / 2, -0.15,
         r'Condition: $\sum_{i \in I_p} M_{ij} = \sum_{i \in I_p} M_{i\ell}$ for all $j, \ell \in I_q$',
         fontsize=10, ha='center', va='top', 
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

ax2.set_xlim(-0.2, 4.2)
ax2.set_ylim(-0.55, 3.5)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title(r'(b) Jacobian block structure', fontsize=11, pad=10)

plt.tight_layout()
plt.savefig(OUT_DIR / 'proper_lumping_blocks.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(OUT_DIR / 'proper_lumping_blocks.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Proper lumping block structure figure created!")
