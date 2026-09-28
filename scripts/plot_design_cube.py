"""
Generate a 3D cube visualization of the 2x2x2 factorial design space.

Axes:
  x = Granularity (Scalar → Channel-wise)
  y = Conditioning (Data-dependent → Data-independent)
  z = Delta rule (No → Yes)

Each vertex is a variant. The missing cell (channel-wise, DD, no-delta)
is shown as a hollow marker.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import numpy as np

fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_subplot(111, projection='3d')

# Vertices: (granularity, conditioning, delta)
# 0 = scalar/DD/no-delta, 1 = channel/DI/delta
variants = {
    # (gran, cond, delta): (name, implemented)
    (0, 0, 0): ("GLA", True),
    (1, 0, 0): ("(RWKV)", False),  # missing cell
    (0, 1, 0): ("ScalarStatic", True),
    (1, 1, 0): ("StaticChannel", True),
    (0, 0, 1): ("DeltaNet", True),
    (1, 0, 1): ("KDA", True),
    (0, 1, 1): ("ScalarStaticDelta", True),
    (1, 1, 1): ("StaticChannelDelta", True),
}

# Draw cube edges
edges = [
    # Bottom face (no delta)
    [(0,0,0), (1,0,0)], [(0,0,0), (0,1,0)],
    [(1,0,0), (1,1,0)], [(0,1,0), (1,1,0)],
    # Top face (delta)
    [(0,0,1), (1,0,1)], [(0,0,1), (0,1,1)],
    [(1,0,1), (1,1,1)], [(0,1,1), (1,1,1)],
    # Vertical edges (connecting no-delta to delta)
    [(0,0,0), (0,0,1)], [(1,0,0), (1,0,1)],
    [(0,1,0), (0,1,1)], [(1,1,0), (1,1,1)],
]

for (x1, y1, z1), (x2, y2, z2) in edges:
    ax.plot([x1, x2], [y1, y2], [z1, z2],
            color='#888888', linewidth=1.0, alpha=0.6, zorder=1)

# Plot vertices
for (g, c, d), (name, impl) in variants.items():
    if name == "StaticChannelDelta":
        ax.scatter(g, c, d, s=120, c='#1f77b4', edgecolors='black',
                   linewidths=1.5, zorder=5, marker='D')
    elif not impl:
        ax.scatter(g, c, d, s=90, c='white', edgecolors='#d62728',
                   linewidths=2, zorder=5, marker='o')
    elif d == 1:  # delta variants
        ax.scatter(g, c, d, s=70, c='#1f77b4', edgecolors='black',
                   linewidths=0.8, zorder=5, marker='o')
    else:  # non-delta
        ax.scatter(g, c, d, s=70, c='#aaaaaa', edgecolors='black',
                   linewidths=0.8, zorder=5, marker='o')

# Label offsets (manually tuned for readability)
label_offsets = {
    (0, 0, 0): (-0.22, -0.08, -0.08),
    (1, 0, 0): (0.08, -0.08, -0.08),
    (0, 1, 0): (-0.30, 0.08, -0.06),
    (1, 1, 0): (0.08, 0.08, -0.06),
    (0, 0, 1): (-0.22, -0.08, 0.06),
    (1, 0, 1): (0.08, -0.08, 0.06),
    (0, 1, 1): (-0.38, 0.08, 0.06),
    (1, 1, 1): (0.08, 0.08, 0.06),
}

for (g, c, d), (name, impl) in variants.items():
    dx, dy, dz = label_offsets[(g, c, d)]
    color = '#d62728' if not impl else ('black' if name != 'StaticChannelDelta' else '#1f77b4')
    weight = 'bold' if name == 'StaticChannelDelta' else 'normal'
    style = 'italic' if not impl else 'normal'
    fontsize = 9.5 if name == 'StaticChannelDelta' else 8.5
    ax.text(g + dx, c + dy, d + dz, name, fontsize=fontsize,
            color=color, fontweight=weight, fontstyle=style,
            ha='center', va='center', zorder=10)

# Axis labels
ax.set_xlabel('Granularity', fontsize=10, labelpad=8)
ax.set_ylabel('Conditioning', fontsize=10, labelpad=8)
ax.set_zlabel('Delta Rule', fontsize=10, labelpad=8)

# Custom tick labels
ax.set_xticks([0, 1])
ax.set_xticklabels(['Scalar', 'Channel'], fontsize=8)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Data-dep', 'Data-indep'], fontsize=8)
ax.set_zticks([0, 1])
ax.set_zticklabels(['No', 'Yes'], fontsize=8)

# Set viewing angle
ax.view_init(elev=22, azim=-55)

# Extend limits slightly for label room
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.2, 1.3)
ax.set_zlim(-0.15, 1.2)

# Clean up the pane
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('lightgray')
ax.yaxis.pane.set_edgecolor('lightgray')
ax.zaxis.pane.set_edgecolor('lightgray')
ax.grid(True, alpha=0.2)

plt.tight_layout()
out_path = 'results/visualizations/design_cube.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")

# Also save to paper figures directory
import shutil
for dest in ['paper/figures/design_cube.png', 'paper/tmlr-style-file-main/figures/design_cube.png']:
    shutil.copy(out_path, dest)
    print(f"Copied to: {dest}")
