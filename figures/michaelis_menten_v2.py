#!/usr/bin/env python3
"""
Figure: Michaelis-Menten reaction network diagram - Publication quality for SIAM.
Clean mathematical style.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "generated"
OUT_DIR.mkdir(exist_ok=True)
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

# Set up figure with publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.size': 11,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8,
})

fig, ax = plt.subplots(figsize=(6.5, 2.8))

# Define positions for species (spread out more)
positions = {
    'S': (0.8, 0.5),
    'E1': (1.6, 0.5),
    'C': (3.2, 0.5),
    'E2': (4.8, 0.5),
    'P': (5.6, 0.5),
}

# Draw species as simple text (no boxes - cleaner)
species_style = {'fontsize': 14, 'ha': 'center', 'va': 'center', 'fontweight': 'bold'}

ax.text(*positions['S'], r'$S$', **species_style)
ax.text(*positions['E1'], r'$E$', **species_style)
ax.text(*positions['C'], r'$C$', **species_style)
ax.text(*positions['E2'], r'$E$', **species_style)
ax.text(*positions['P'], r'$P$', **species_style)

# Plus signs
ax.text(1.2, 0.5, '+', fontsize=14, ha='center', va='center')
ax.text(5.2, 0.5, '+', fontsize=14, ha='center', va='center')

# Draw reaction arrows
arrow_style = dict(arrowstyle='->', color='black', lw=1.2, 
                   mutation_scale=12, shrinkA=8, shrinkB=8)
arrow_style_rev = dict(arrowstyle='<-', color='black', lw=1.2,
                       mutation_scale=12, shrinkA=8, shrinkB=8)

# First reaction: S + E <-> C
# Forward arrow (top)
ax.annotate('', xy=(2.95, 0.58), xytext=(1.85, 0.58), arrowprops=arrow_style)
# Reverse arrow (bottom)  
ax.annotate('', xy=(1.85, 0.42), xytext=(2.95, 0.42), arrowprops=arrow_style)

# Rate labels for first reaction
ax.text(2.4, 0.75, r'$k_1$', fontsize=11, ha='center', va='bottom')
ax.text(2.4, 0.25, r'$k_{-1}$', fontsize=11, ha='center', va='top')

# Second reaction: C <-> E + P
# Forward arrow (top)
ax.annotate('', xy=(4.55, 0.58), xytext=(3.45, 0.58), arrowprops=arrow_style)
# Reverse arrow (bottom)
ax.annotate('', xy=(3.45, 0.42), xytext=(4.55, 0.42), arrowprops=arrow_style)

# Rate labels for second reaction
ax.text(4.0, 0.75, r'$k_2$', fontsize=11, ha='center', va='bottom')
ax.text(4.0, 0.25, r'$k_{-2}$', fontsize=11, ha='center', va='top')

# Critical parameter box (simple rectangle)
box_x, box_y = 2.4, -0.25
box_w, box_h = 1.6, 0.35
rect = Rectangle((box_x, box_y), box_w, box_h, 
                  fill=False, edgecolor='black', linewidth=1.0, linestyle='-')
ax.add_patch(rect)

ax.text(box_x + box_w/2, box_y + box_h/2, r'Critical: $k_1 = k_{-2}$', 
        fontsize=11, ha='center', va='center')

# Lumping information at top
ax.text(3.2, 0.95, r'Lumping variables: $y_1 = s + p$,  $y_2 = e + c$,  $y_3 = s + c + p$',
        fontsize=10, ha='center', va='bottom', style='italic')

# Clean up axes
ax.set_xlim(0.3, 6.1)
ax.set_ylim(-0.35, 1.15)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig(OUT_DIR / 'michaelis_menten_network.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(OUT_DIR / 'michaelis_menten_network.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Publication-quality Michaelis-Menten figure created!")
