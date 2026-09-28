"""
Clean isometric cube for the 2x2x2 factorial design space.

Axes are labeled along separated edges to avoid overlap:
  - Granularity: bottom-front edge (horizontal)
  - Delta Rule: left-front edge (vertical)
  - Conditioning: right-back edge (diagonal/depth)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def project(x, y, z):
    """Isometric projection.
    x = Granularity  (Scalar=0, Channel=1)
    y = Conditioning (DD=0, DI=1)
    z = Delta rule   (No=0, Yes=1)
    """
    px = 3.0 * x + 1.0 * y
    py = 0.65 * y + 2.8 * z
    return px, py

# Variants
vertices = [
    (0,0,0, "GLA",                True,  False),
    (1,0,0, "(RWKV)",             False, False),
    (0,1,0, "ScalarStatic",       True,  False),
    (1,1,0, "StaticChannel",      True,  False),
    (0,0,1, "DeltaNet",           True,  False),
    (1,0,1, "KDA",                True,  False),
    (0,1,1, "ScalarStaticDelta",  True,  False),
    (1,1,1, "StaticChannelDelta", True,  True),
]

edges_back = [((0,0,0),(0,1,0)), ((0,0,0),(1,0,0)), ((0,0,0),(0,0,1))]
edges_front = [
    ((1,0,0),(1,1,0)), ((0,1,0),(1,1,0)),
    ((1,0,0),(1,0,1)), ((0,1,0),(0,1,1)),
    ((0,0,1),(1,0,1)), ((0,0,1),(0,1,1)),
    ((1,1,0),(1,1,1)), ((1,0,1),(1,1,1)), ((0,1,1),(1,1,1)),
]

# Node label positions: (dx, dy, ha, va)
labels = {
    "GLA":                (-0.12, -0.28, "center", "top"),
    "(RWKV)":             ( 0.00, -0.32, "center", "top"),
    "ScalarStatic":       (-0.15,  0.05, "right",  "center"),   # left of node (edges go right, up, down-left)
    "StaticChannel":      (-0.15,  0.15, "right",  "bottom"),  # above-left of node (edges go left, up, down-left)
    "DeltaNet":           (-0.20,  0.20, "right",  "bottom"),
    "KDA":                ( 0.15, -0.12, "left",   "top"),     # right-below node (edges go left, up-right, down)
    "ScalarStaticDelta":  (-0.20,  0.20, "right",  "bottom"),
    "StaticChannelDelta": ( 0.22,  0.25, "left",   "bottom"),
}

C_D  = "#2166ac"   # delta blue
C_ND = "#878787"   # non-delta gray
C_M  = "#c0392b"   # missing red
C_E  = "#555555"   # front edge
C_EB = "#b0b0b0"   # back edge

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.set_aspect('equal')
ax.axis('off')

# Shade faces
bot = [project(0,0,0), project(1,0,0), project(1,1,0), project(0,1,0)]
top = [project(0,0,1), project(1,0,1), project(1,1,1), project(0,1,1)]
ax.add_patch(plt.Polygon(bot, fc='#eeeeee', ec='none', alpha=0.5, zorder=0))
ax.add_patch(plt.Polygon(top, fc='#d1e5f0', ec='none', alpha=0.5, zorder=0))

# Face labels
bot_cx = np.mean([p[0] for p in bot])
bot_cy = np.mean([p[1] for p in bot])
top_cx = np.mean([p[0] for p in top])
top_cy = np.mean([p[1] for p in top])
ax.text(bot_cx, bot_cy, 'No Delta', fontsize=9, ha='center', va='center',
        color='#999999', fontstyle='italic', zorder=0.5)
ax.text(top_cx, top_cy, 'Delta', fontsize=9, ha='center', va='center',
        color='#7faed1', fontstyle='italic', zorder=0.5)

# Edges
for a, b in edges_back:
    p1, p2 = project(*a), project(*b)
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], color=C_EB, lw=1.0, ls='--', zorder=1)
for a, b in edges_front:
    p1, p2 = project(*a), project(*b)
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], color=C_E, lw=1.2, ls='-', zorder=2)

# Vertices
for x, y, z, name, impl, hl in vertices:
    px, py = project(x, y, z)
    dx, dy, ha, va = labels[name]
    if hl:
        ax.plot(px, py, 'D', ms=13, color=C_D, mec='black', mew=1.5, zorder=5)
        ax.text(px+dx, py+dy, name, fontsize=10.5, fontweight='bold',
                color=C_D, ha=ha, va=va, zorder=6)
    elif not impl:
        ax.plot(px, py, 'o', ms=9.5, color='white', mec=C_M, mew=2.0, zorder=5)
        ax.text(px+dx, py+dy, name, fontsize=8.5, fontstyle='italic',
                color=C_M, ha=ha, va=va, zorder=6)
    elif z == 1:
        ax.plot(px, py, 'o', ms=8.5, color=C_D, mec='black', mew=0.8, zorder=5)
        ax.text(px+dx, py+dy, name, fontsize=8.5, color='#222', ha=ha, va=va, zorder=6)
    else:
        ax.plot(px, py, 'o', ms=8.5, color=C_ND, mec='black', mew=0.8, zorder=5)
        ax.text(px+dx, py+dy, name, fontsize=8.5, color='#222', ha=ha, va=va, zorder=6)

# ===== AXIS ANNOTATIONS =====

# 1) GRANULARITY: arrow below the bottom-front edge
p0 = project(0, 0, 0)
p1 = project(1, 0, 0)
arr_y = p0[1] - 0.75
ax.annotate('', xy=(p1[0], arr_y), xytext=(p0[0], arr_y),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text((p0[0]+p1[0])/2, arr_y - 0.15, 'Granularity',
        fontsize=11, fontweight='bold', ha='center', va='top')
ax.text(p0[0], arr_y + 0.10, 'Scalar', fontsize=8.5, ha='center', va='bottom', color='#444')
ax.text(p1[0], arr_y + 0.10, 'Channel', fontsize=8.5, ha='center', va='bottom', color='#444')

# 2) DELTA RULE: arrow to the left of the left-front vertical edge
p_bot = project(0, 0, 0)
p_top = project(0, 0, 1)
arr_x = p_bot[0] - 0.55
ax.annotate('', xy=(arr_x, p_top[1]), xytext=(arr_x, p_bot[1]),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(arr_x - 0.50, (p_bot[1]+p_top[1])/2, 'Delta\nRule',
        fontsize=11, fontweight='bold', ha='center', va='center')
ax.text(arr_x - 0.12, p_bot[1], 'No', fontsize=8.5, ha='right', va='center', color='#444')
ax.text(arr_x - 0.12, p_top[1], 'Yes', fontsize=8.5, ha='right', va='center', color='#444')

# 3) CONDITIONING: arrow to the right of the right-bottom depth edge, pushed further out
p_dd = project(1, 0, 0)
p_di = project(1, 1, 0)
off = 0.85  # push far enough right to clear StaticChannel label
p_dd_r = (p_dd[0] + off, p_dd[1])
p_di_r = (p_di[0] + off, p_di[1])
ax.annotate('', xy=p_di_r, xytext=p_dd_r,
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
angle = np.degrees(np.arctan2(p_di_r[1]-p_dd_r[1], p_di_r[0]-p_dd_r[0]))
mid_r = ((p_dd_r[0]+p_di_r[0])/2 + 0.45, (p_dd_r[1]+p_di_r[1])/2)
ax.text(mid_r[0], mid_r[1], 'Conditioning', fontsize=11, fontweight='bold',
        ha='center', va='center', rotation=angle)
ax.text(p_dd_r[0]+0.10, p_dd_r[1]-0.12, 'Data-dep',
        fontsize=7.5, ha='left', va='top', color='#444')
ax.text(p_di_r[0]+0.10, p_di_r[1]+0.12, 'Data-indep',
        fontsize=7.5, ha='left', va='bottom', color='#444')

# Legend - place in upper left to avoid bottom-right crowding
leg = [
    plt.Line2D([0],[0], marker='o', color='w', mfc=C_D, mec='black', ms=8, label='Delta variant'),
    plt.Line2D([0],[0], marker='o', color='w', mfc=C_ND, mec='black', ms=8, label='Non-delta variant'),
    plt.Line2D([0],[0], marker='o', color='w', mfc='white', mec=C_M, mew=2, ms=8, label='Not implemented'),
    plt.Line2D([0],[0], marker='D', color='w', mfc=C_D, mec='black', ms=9, label='StaticChannelDelta (ours)'),
]
ax.legend(handles=leg, loc='lower center', fontsize=8.5,
          framealpha=0.95, edgecolor='#ccc', borderpad=0.8,
          bbox_to_anchor=(0.5, -0.06), ncol=2)

ax.set_xlim(-1.8, 6.5)
ax.set_ylim(-2.0, 4.5)

plt.savefig('results/visualizations/design_cube_v2.png',
            dpi=250, bbox_inches='tight', facecolor='white', pad_inches=0.12)
print("Saved: results/visualizations/design_cube_v2.png")

import shutil
for d in ['paper/figures/design_cube.png', 'paper/tmlr-style-file-main/figures/design_cube.png']:
    shutil.copy('results/visualizations/design_cube_v2.png', d)
    print(f"Copied to: {d}")
