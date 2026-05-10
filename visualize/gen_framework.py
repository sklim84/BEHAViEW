"""BehaView framework figure (matplotlib).

Comprehensive design including:
- Method name in title
- Two-branch (online/target) BYOL structure
- EMA momentum arrow (top -> bottom encoder)
- Stop-gradient symbol on target output
- Predictor q on online side; projector g on both sides
- Subgraph pool annotated with respective neighborhood graph (tx vs k-NN)
- Bootstrap loss with cross-view symmetry
- Inference path with concat + LogReg
- 4-setting legend with (d) star marker
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = '/home/work/kftc_model/KA-003-FraudCenGCL/_paper/figures'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# ============ Color palette ============
C_INPUT     = '#E1F5FE'
C_VIEW1     = '#BBDEFB'
C_VIEW2     = '#FFE0B2'
C_ENC_ON    = '#A5D6A7'   # online encoder (slightly brighter green)
C_ENC_TG    = '#C8E6C9'   # target encoder (lighter)
C_POOL_TX   = '#CE93D8'   # subgraph pool over tx graph
C_POOL_KNN  = '#E1BEE7'   # subgraph pool over k-NN graph
C_PROJ      = '#F8BBD0'
C_LOSS      = '#FFCDD2'
C_INFER     = '#FFF59D'
C_PROPOSED  = '#FF8A65'
C_BORDER    = '#37474F'
C_TEXT      = '#212121'
C_SUB       = '#546E7A'
C_AXIS      = '#1565C0'
C_ARROW     = '#455A64'
C_EMA       = '#7B1FA2'
C_SG        = '#C62828'

fig = plt.figure(figsize=(13.5, 5.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 27)
ax.set_ylim(0, 11)
ax.axis('off')

# ---------- Helpers ----------
def box(x, y, w, h, color, edge=C_BORDER, lw=1.0, **kw):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                       facecolor=color, edgecolor=edge, linewidth=lw, **kw)
    ax.add_patch(p)

def text(x, y, t, fs=9, weight='normal', color=C_TEXT, ha='center', va='center', style='normal'):
    ax.text(x, y, t, fontsize=fs, fontweight=weight, color=color, ha=ha, va=va, style=style)

def arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.3, style='->', text_label=None,
          text_offset=(0, 0.25), text_color=None, text_size=7.5):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, color=color, linewidth=lw,
                        mutation_scale=12, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if text_label:
        ax.text((x1+x2)/2 + text_offset[0], (y1+y2)/2 + text_offset[1],
                text_label, fontsize=text_size, color=text_color or color,
                ha='center', va='center', style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.85))

# ============ Title (top) ============
text(13.5, 10.45,
     '$\\mathtt{BehaView}$: Behavioral Subgraph Contrast Framework',
     fs=12.5, weight='bold')

# ============ Layout grid ============
# 6 columns: Input | View | Encoder | Pool | Projection/Predictor | Loss/Inference
COL = {
    'IN':   (1.0,  2.4),   # x_left, width
    'VIEW': (4.4,  3.2),
    'ENC':  (8.5,  2.8),
    'POOL': (12.3, 3.0),
    'PROJ': (16.4, 2.8),
    'LOSS': (20.3, 2.6),
    'INFR': (23.5, 3.0),
}
ROW_T_Y = 6.6  # online row (top)
ROW_B_Y = 2.5  # target row (bottom)
ROW_H = 1.7

# ============ Input (single, branches into top + bottom) ============
ix, iw = COL['IN']
icx = ix + iw/2
iy = (ROW_T_Y + ROW_B_Y) / 2 - 0.6
ih = 1.5
box(ix, iy, iw, ih, C_INPUT)
text(icx, iy + 1.05, 'Transaction', fs=10, weight='bold')
text(icx, iy + 0.55, 'Graph', fs=10, weight='bold')
text(icx, iy + 0.10, '$\\mathcal{G}_{\\mathrm{tx}}=(V,E,X)$', fs=8.5, color=C_SUB)

# ============ View 1 (top) ============
vx, vw = COL['VIEW']
vcx = vx + vw/2
box(vx, ROW_T_Y, vw, ROW_H, C_VIEW1)
text(vcx, ROW_T_Y + 1.40, 'View 1: Augmented', fs=10, weight='bold')
text(vcx, ROW_T_Y + 1.00, 'Edge drop ($p_e{=}0.3$)', fs=8.2, color=C_SUB)
text(vcx, ROW_T_Y + 0.65, 'Feat mask ($p_f{=}0.3$)', fs=8.2, color=C_SUB)
text(vcx, ROW_T_Y + 0.20, 'S-S 15% : S-B 85%', fs=8.5, weight='bold', color='#C62828')

# ============ View 2 (bottom) ============
box(vx, ROW_B_Y, vw, ROW_H, C_VIEW2)
text(vcx, ROW_B_Y + 1.40, 'View 2: Behav. k-NN', fs=10, weight='bold')
text(vcx, ROW_B_Y + 1.00, '$\\bar{x}_v$: standardize+L2', fs=8.2, color=C_SUB)
text(vcx, ROW_B_Y + 0.65, 'ball-tree, $k{=}10$', fs=8.2, color=C_SUB)
text(vcx, ROW_B_Y + 0.20, 'S-S 41% : S-B 59%', fs=8.5, weight='bold', color='#2E7D32')

# Input → views
arrow(ix+iw, iy + ih*0.7, vx, ROW_T_Y + ROW_H/2)
arrow(ix+iw, iy + ih*0.3, vx, ROW_B_Y + ROW_H/2)

# Axis 1 label
text(vcx, (ROW_T_Y + ROW_B_Y + ROW_H)/2, 'View Construction $\\updownarrow$',
     fs=8.5, weight='bold', color=C_AXIS)

# ============ Encoder ============
ex, ew = COL['ENC']
ecx = ex + ew/2
box(ex, ROW_T_Y, ew, ROW_H, C_ENC_ON)
text(ecx, ROW_T_Y + 1.35, 'Online $f_\\theta$', fs=10, weight='bold')
text(ecx, ROW_T_Y + 0.85, 'GCNConv + BN', fs=8.5, color=C_SUB)
text(ecx, ROW_T_Y + 0.55, '+ PReLU + Dropout', fs=8.5, color=C_SUB)
text(ecx, ROW_T_Y + 0.18, '$L{=}2$, $d{=}256$', fs=8.0, color=C_SUB, weight='normal')

box(ex, ROW_B_Y, ew, ROW_H, C_ENC_TG)
text(ecx, ROW_B_Y + 1.35, 'Target $f_\\xi$', fs=10, weight='bold')
text(ecx, ROW_B_Y + 0.85, 'GCNConv + BN', fs=8.5, color=C_SUB)
text(ecx, ROW_B_Y + 0.55, '+ PReLU + Dropout', fs=8.5, color=C_SUB)
text(ecx, ROW_B_Y + 0.18, '(no gradient)', fs=8.0, color=C_SG, weight='normal')

# View → Encoder
arrow(vx+vw, ROW_T_Y + ROW_H/2, ex, ROW_T_Y + ROW_H/2)
arrow(vx+vw, ROW_B_Y + ROW_H/2, ex, ROW_B_Y + ROW_H/2)

# EMA arrow online -> target (vertical)
arrow(ecx - 0.2, ROW_T_Y, ecx - 0.2, ROW_B_Y + ROW_H,
      color=C_EMA, lw=1.4, style='->',
      text_label='EMA\n$m{=}0.99$', text_offset=(-0.95, 0), text_color=C_EMA, text_size=7.5)

# ============ Subgraph Pool ============
px, pw = COL['POOL']
pcx = px + pw/2
box(px, ROW_T_Y, pw, ROW_H, C_POOL_TX)
text(pcx, ROW_T_Y + 1.35, 'Subgraph Pool', fs=10, weight='bold')
text(pcx, ROW_T_Y + 0.85, '$s^{(1)}_v = \\mathrm{mean}(h_v \\cup \\mathcal{N}_{\\mathrm{tx}}(v))$',
     fs=8.0, color=C_SUB)
text(pcx, ROW_T_Y + 0.30, 'tx-graph neighborhood', fs=7.8, color='#6A1B9A', style='italic')

box(px, ROW_B_Y, pw, ROW_H, C_POOL_KNN)
text(pcx, ROW_B_Y + 1.35, 'Subgraph Pool', fs=10, weight='bold')
text(pcx, ROW_B_Y + 0.85, '$s^{(2)}_v = \\mathrm{mean}(h_v \\cup \\mathcal{N}_{\\mathrm{knn}}(v))$',
     fs=8.0, color=C_SUB)
text(pcx, ROW_B_Y + 0.30, 'k-NN neighborhood', fs=7.8, color='#6A1B9A', style='italic')

# Encoder → Pool
arrow(ex+ew, ROW_T_Y + ROW_H/2, px, ROW_T_Y + ROW_H/2,
      text_label='$h^{(1)}$', text_offset=(0, 0.32), text_size=8)
arrow(ex+ew, ROW_B_Y + ROW_H/2, px, ROW_B_Y + ROW_H/2,
      text_label='$h^{(2)}$', text_offset=(0, 0.32), text_size=8)

# Axis 2 label
text(pcx, (ROW_T_Y + ROW_B_Y + ROW_H)/2, 'Contrastive Level $\\updownarrow$',
     fs=8.5, weight='bold', color=C_AXIS)

# ============ Projection & Predictor ============
prx, prw = COL['PROJ']
prcx = prx + prw/2
# Top: g_θ + q_θ
box(prx, ROW_T_Y, prw, ROW_H, C_PROJ)
text(prcx, ROW_T_Y + 1.35, 'Project $g_\\theta$', fs=10, weight='bold')
text(prcx, ROW_T_Y + 0.85, '+ Predict $q_\\theta$', fs=10, weight='bold')
text(prcx, ROW_T_Y + 0.32, '$\\hat{y}^{(1)} = q(g(s^{(1)}))$', fs=8.0, color=C_SUB)

# Bottom: g_ξ only
box(prx, ROW_B_Y, prw, ROW_H, C_PROJ)
text(prcx, ROW_B_Y + 1.35, 'Project $g_\\xi$', fs=10, weight='bold')
text(prcx, ROW_B_Y + 0.85, '(no predictor)', fs=8.5, color=C_SUB, style='italic')
text(prcx, ROW_B_Y + 0.32, '$z^{(2)} = \\mathrm{sg}(g(s^{(2)}))$', fs=8.0, color=C_SG)

# Pool → Proj
arrow(px+pw, ROW_T_Y + ROW_H/2, prx, ROW_T_Y + ROW_H/2)
arrow(px+pw, ROW_B_Y + ROW_H/2, prx, ROW_B_Y + ROW_H/2)

# ============ BYOL Loss (centered between rows) ============
lx, lw = COL['LOSS']
lcx = lx + lw/2
ly = (ROW_T_Y + ROW_B_Y + ROW_H)/2 - 1.0
lh = 2.0
box(lx, ly, lw, lh, C_LOSS, lw=1.4)
text(lcx, ly + 1.65, 'BYOL', fs=10.5, weight='bold')
text(lcx, ly + 1.30, 'Bootstrap Loss', fs=10.5, weight='bold')
text(lcx, ly + 0.80, 'symmetric', fs=8.0, color=C_SUB, style='italic')
text(lcx, ly + 0.40, '$\\mathcal{L}=\\sum_k 2{-}2\\cdot$', fs=8.5, color=C_TEXT)
text(lcx, ly + 0.05, '$\\cos(\\hat{y}^{(k)}, z^{(3-k)})$', fs=8.5, color=C_TEXT)

# Proj → Loss (curve from top and bottom)
arrow(prx+prw, ROW_T_Y + ROW_H/2, lx, ly + lh*0.78)
arrow(prx+prw, ROW_B_Y + ROW_H/2, lx, ly + lh*0.22)

# ============ Inference path (right, separate) ============
inx, inw = COL['INFR']
incx = inx + inw/2
iny = (ROW_T_Y + ROW_B_Y + ROW_H)/2 - 1.4
inh = 2.8
box(inx, iny, inw, inh, C_INFER, lw=1.2)
text(incx, iny + 2.40, 'Inference', fs=10.5, weight='bold')
text(incx, iny + 1.95, '(frozen $f_\\theta$)', fs=8, color=C_SUB, style='italic')
text(incx, iny + 1.45, '$z_v = [s^{(1)}_v \\| s^{(2)}_v]$', fs=9, color=C_TEXT)
text(incx, iny + 1.00, '$\\downarrow$', fs=11, color=C_ARROW)
text(incx, iny + 0.65, 'LogReg', fs=10, weight='bold')
text(incx, iny + 0.20, 'F1\\_susp / AUROC', fs=8.5, color='#C62828', weight='bold')

# Loss → Inference (skip arrow, dashed)
arrow(lx+lw, ly + lh/2, inx, iny + inh/2,
      style='->', lw=0.9, color=C_SUB)

# ============ Bottom legend (4-setting) ============
leg_y = 0.4
leg_x = 1.0
leg_w = 25.0
leg_h = 1.05
box(leg_x, leg_y, leg_w, leg_h, '#FAFAFA', edge='#BDBDBD', lw=0.8)

text(leg_x + 0.3, leg_y + leg_h/2, '4 settings:',
     fs=9, weight='bold', ha='left')

settings_x_offset = 3.0
settings = [
    ('(a)', 'Aug. View + Node-level',     C_VIEW1, False),
    ('(b)', 'Behav. k-NN + Node-level',   C_VIEW2, False),
    ('(c)', 'Aug. View + Subgraph Pool',  C_POOL_TX, False),
    ('(d)', 'Behav. k-NN + Subgraph Pool ($\\star$ proposed)', C_PROPOSED, True),
]
slot_w = (leg_w - settings_x_offset - 0.6) / 4
for i, (tag, desc, col, star) in enumerate(settings):
    x0 = leg_x + settings_x_offset + i * slot_w
    sw = 0.4
    sh = 0.5
    sy = leg_y + (leg_h - sh)/2
    box(x0, sy, sw, sh, col,
        edge=C_PROPOSED if star else C_BORDER,
        lw=1.6 if star else 0.8)
    text(x0 + sw + 0.15, leg_y + leg_h/2, tag,
         fs=9, weight='bold', ha='left')
    text(x0 + sw + 0.65, leg_y + leg_h/2, desc,
         fs=8.0, ha='left', color=C_SUB)

plt.savefig(os.path.join(OUT, 'fig_framework.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT, 'fig_framework.png'), bbox_inches='tight', dpi=300)
plt.close()
print('Saved:', os.path.join(OUT, 'fig_framework.pdf'))
