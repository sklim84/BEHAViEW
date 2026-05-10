"""BehaView framework figure (matplotlib).

Uniform-width design:
- All pipeline boxes share the same width (BW)
- Transaction Graph (start), BYOL Loss, and Inference (end) share the same
  width AND height (TERMINAL_W x TERMINAL_H)
- Text inside each box reduced to component name + role only
- Hyperparameters/equations moved to caption and main text
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = '/home/work/kftc_model/KA-003-FraudCenGCL/_paper/figures'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# ============ Color palette (simplified to 3 functional categories) ============
# Terminal boxes (Input, Loss, Inference) — cyan
# View 1 (Augmented) — blue, matches legend (a)/(c)
# View 2 (Behavioral k-NN) — orange, matches legend (b)/(d)
# All other process boxes (Encoder, Pool, Projector) — neutral gray
C_TERMINAL  = '#E1F5FE'   # cyan: terminal endpoints (Input, Loss, Inference)
C_VIEW1     = '#BBDEFB'   # blue: View 1 (Augmented)
C_VIEW2     = '#FFE0B2'   # orange: View 2 (Behavioral k-NN)
C_PROCESS   = '#ECEFF1'   # light gray: Encoder, Pool, Projector
C_PROPOSED  = '#FF8A65'   # coral: legend (d) star marker only
C_BORDER    = '#37474F'
C_TEXT      = '#212121'
C_SUB       = '#546E7A'
C_AXIS      = '#1565C0'
C_ARROW     = '#455A64'
C_EMA       = '#7B1FA2'
C_SG        = '#C62828'

# ============ Layout grid ============
# All pipeline boxes have uniform width BW.
# Terminal boxes (Input, BYOL Loss, Inference) share width = TERMINAL_W and height = TERMINAL_H.
BW = 3.0           # uniform pipeline box width
ROW_H = 1.7        # height of each row box (View/Encoder/Pool/Proj)
TERMINAL_W = 3.0   # input + BYOL + Inference width (== BW for visual unity)
TERMINAL_H = 4.0   # input + BYOL + Inference height (spans both rows)

GAP = 0.85         # horizontal gap between columns

# Column x positions (left edge of each box)
x_in   = 0.5
x_v    = x_in + TERMINAL_W + GAP
x_enc  = x_v   + BW + GAP
x_pool = x_enc + BW + GAP
x_proj = x_pool + BW + GAP
x_loss = x_proj + BW + GAP
x_infr = x_loss + TERMINAL_W + GAP

ROW_T_Y = 6.0   # online row (top) bottom-edge y
ROW_B_Y = 1.7   # target row (bottom) bottom-edge y

# Terminal box vertical placement: spans both rows (centered)
TERMINAL_Y = (ROW_T_Y + ROW_B_Y + ROW_H - TERMINAL_H) / 2

# Total figure dimensions
TOTAL_X = x_infr + TERMINAL_W + 0.5
TOTAL_Y = 11.0

fig = plt.figure(figsize=(14.0, 5.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, TOTAL_X)
ax.set_ylim(0, TOTAL_Y)
ax.axis('off')

# ---------- Helpers ----------
def box(x, y, w, h, color, edge=C_BORDER, lw=1.0, **kw):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                       facecolor=color, edgecolor=edge, linewidth=lw, **kw)
    ax.add_patch(p)

def text(x, y, t, fs=11, weight='normal', color=C_TEXT, ha='center', va='center', style='normal'):
    ax.text(x, y, t, fontsize=fs, fontweight=weight, color=color, ha=ha, va=va, style=style)

def arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.4, style='->',
          text_label=None, text_offset=(0, 0.3), text_color=None, text_size=8.5):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, color=color, linewidth=lw,
                        mutation_scale=14, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if text_label:
        ax.text((x1+x2)/2 + text_offset[0], (y1+y2)/2 + text_offset[1],
                text_label, fontsize=text_size, color=text_color or color,
                ha='center', va='center', style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.9))

# Title removed — caption already conveys 'Overview of \method'.

# ============ Input (terminal) ============
icx = x_in + TERMINAL_W/2
icy = TERMINAL_Y + TERMINAL_H/2
box(x_in, TERMINAL_Y, TERMINAL_W, TERMINAL_H, C_TERMINAL, lw=1.3)
text(icx, icy + 0.35, 'Transaction', fs=12, weight='bold')
text(icx, icy - 0.15, 'Graph', fs=12, weight='bold')
text(icx, icy - 0.65, '$\\mathcal{G}_{\\mathrm{tx}}$', fs=11, color=C_SUB)

# ============ View 1 / View 2 ============
vcx = x_v + BW/2
box(x_v, ROW_T_Y, BW, ROW_H, C_VIEW1)
text(vcx, ROW_T_Y + ROW_H*0.65, 'View 1', fs=11.5, weight='bold')
text(vcx, ROW_T_Y + ROW_H*0.30, 'Augmented Graph', fs=10, color=C_SUB)

box(x_v, ROW_B_Y, BW, ROW_H, C_VIEW2)
text(vcx, ROW_B_Y + ROW_H*0.65, 'View 2', fs=11.5, weight='bold')
text(vcx, ROW_B_Y + ROW_H*0.30, 'Behavioral k-NN', fs=10, color=C_SUB)

# Input → views
arrow(x_in + TERMINAL_W, icy + 0.4, x_v, ROW_T_Y + ROW_H/2)
arrow(x_in + TERMINAL_W, icy - 0.4, x_v, ROW_B_Y + ROW_H/2)

# Axis 1
text(vcx, (ROW_T_Y + ROW_B_Y + ROW_H)/2,
     'View Construction $\\updownarrow$', fs=11.5, weight='bold', color=C_AXIS)

# ============ Encoder ============
ecx = x_enc + BW/2
box(x_enc, ROW_T_Y, BW, ROW_H, C_PROCESS)
text(ecx, ROW_T_Y + ROW_H*0.62, 'Online Encoder', fs=11.5, weight='bold')
text(ecx, ROW_T_Y + ROW_H*0.30, '$f_\\theta$  (GCN+BN)', fs=10, color=C_SUB)

box(x_enc, ROW_B_Y, BW, ROW_H, C_PROCESS)
text(ecx, ROW_B_Y + ROW_H*0.62, 'Target Encoder', fs=11.5, weight='bold')
text(ecx, ROW_B_Y + ROW_H*0.30, '$f_\\xi$  (EMA, stop-grad)', fs=10, color=C_SUB)

# View → Encoder
arrow(x_v + BW, ROW_T_Y + ROW_H/2, x_enc, ROW_T_Y + ROW_H/2)
arrow(x_v + BW, ROW_B_Y + ROW_H/2, x_enc, ROW_B_Y + ROW_H/2)

# EMA (online → target, vertical)
arrow(ecx - 0.3, ROW_T_Y, ecx - 0.3, ROW_B_Y + ROW_H,
      color=C_EMA, lw=1.5, text_label='EMA',
      text_offset=(-0.85, 0), text_color=C_EMA, text_size=11)

# ============ Pool ============
pcx = x_pool + BW/2
box(x_pool, ROW_T_Y, BW, ROW_H, C_PROCESS)
text(pcx, ROW_T_Y + ROW_H*0.62, 'Subgraph Pool', fs=11.5, weight='bold')
text(pcx, ROW_T_Y + ROW_H*0.30, 'tx-graph neighbors', fs=10, color=C_SUB)

box(x_pool, ROW_B_Y, BW, ROW_H, C_PROCESS)
text(pcx, ROW_B_Y + ROW_H*0.62, 'Subgraph Pool', fs=11.5, weight='bold')
text(pcx, ROW_B_Y + ROW_H*0.30, 'k-NN neighbors', fs=10, color=C_SUB)

# Encoder → Pool
arrow(x_enc + BW, ROW_T_Y + ROW_H/2, x_pool, ROW_T_Y + ROW_H/2)
arrow(x_enc + BW, ROW_B_Y + ROW_H/2, x_pool, ROW_B_Y + ROW_H/2)

# Axis 2
text(pcx, (ROW_T_Y + ROW_B_Y + ROW_H)/2,
     'Contrastive Level $\\updownarrow$', fs=11.5, weight='bold', color=C_AXIS)

# ============ Projector / Predictor ============
prcx = x_proj + BW/2
box(x_proj, ROW_T_Y, BW, ROW_H, C_PROCESS)
text(prcx, ROW_T_Y + ROW_H*0.62, 'Projector', fs=11.5, weight='bold')
text(prcx, ROW_T_Y + ROW_H*0.30, '$g_\\theta$ + Predictor $q_\\theta$', fs=10, color=C_SUB)

box(x_proj, ROW_B_Y, BW, ROW_H, C_PROCESS)
text(prcx, ROW_B_Y + ROW_H*0.62, 'Projector', fs=11.5, weight='bold')
text(prcx, ROW_B_Y + ROW_H*0.30, '$g_\\xi$  (stop-grad)', fs=10, color=C_SUB, style='italic')

# Pool → Proj
arrow(x_pool + BW, ROW_T_Y + ROW_H/2, x_proj, ROW_T_Y + ROW_H/2)
arrow(x_pool + BW, ROW_B_Y + ROW_H/2, x_proj, ROW_B_Y + ROW_H/2)

# ============ BYOL Loss (terminal) ============
lcx = x_loss + TERMINAL_W/2
lcy = TERMINAL_Y + TERMINAL_H/2
box(x_loss, TERMINAL_Y, TERMINAL_W, TERMINAL_H, C_TERMINAL, lw=1.3)
text(lcx, lcy + 0.5, 'BYOL', fs=12, weight='bold')
text(lcx, lcy + 0.0, 'Bootstrap Loss', fs=12, weight='bold')
text(lcx, lcy - 0.55, 'symmetric, no neg.', fs=9.5, color=C_SUB, style='italic')

# Proj → Loss (converge)
arrow(x_proj + BW, ROW_T_Y + ROW_H/2, x_loss, lcy + 0.6)
arrow(x_proj + BW, ROW_B_Y + ROW_H/2, x_loss, lcy - 0.6)

# ============ Inference (terminal) ============
incx = x_infr + TERMINAL_W/2
incy = TERMINAL_Y + TERMINAL_H/2
box(x_infr, TERMINAL_Y, TERMINAL_W, TERMINAL_H, C_TERMINAL, lw=1.3)
text(incx, incy + 0.6, 'Inference', fs=12, weight='bold')
text(incx, incy + 0.05, '$z_v = [s^{(1)}\\,\\|\\,s^{(2)}]$', fs=10.5, color=C_TEXT)
text(incx, incy - 0.45, '$\\downarrow$ LogReg', fs=10.5, weight='bold')
text(incx, incy - 0.95, '$F1_{\\mathrm{susp}}$ / AUROC', fs=10, color='#C62828', weight='bold')

# Loss → Inference (dashed flow indicator)
arrow(x_loss + TERMINAL_W, lcy, x_infr, incy, lw=1.0, color=C_SUB)

# ============ Bottom legend (4-setting) ============
leg_y = 0.3
leg_h = 0.85
leg_x = 0.5
leg_w = TOTAL_X - 1.0
box(leg_x, leg_y, leg_w, leg_h, '#FAFAFA', edge='#BDBDBD', lw=0.8)

text(leg_x + 0.3, leg_y + leg_h/2, '4 settings:',
     fs=10, weight='bold', ha='left')

settings_x_offset = 2.7
# Legend colors match framework body:
# View 1 (blue) = augmented; View 2 (orange) = behavioral.
# Subgraph variants (c)(d) reuse the same view color, distinguished by
# an additional gray pool indicator stacked next to the view marker.
settings = [
    ('(a)', 'Aug. + Node',     C_VIEW1, False, False),
    ('(b)', 'Behav. + Node',   C_VIEW2, False, False),
    ('(c)', 'Aug. + Subgraph', C_VIEW1, True,  False),
    ('(d)', 'Behav. + Subgraph ($\\star$ proposed)', C_VIEW2, True, True),
]
slot_w = (leg_w - settings_x_offset - 0.6) / 4
for i, (tag, desc, col, has_pool, is_proposed) in enumerate(settings):
    x0 = leg_x + settings_x_offset + i * slot_w
    sw = 0.4
    sh = 0.45
    sy = leg_y + (leg_h - sh)/2
    # Main view-color marker
    box(x0, sy, sw, sh, col,
        edge=C_PROPOSED if is_proposed else C_BORDER,
        lw=1.6 if is_proposed else 0.8)
    # Pool indicator (small gray box adjacent) for (c)/(d)
    if has_pool:
        box(x0 + sw + 0.05, sy, sw*0.55, sh, C_PROCESS,
            edge=C_PROPOSED if is_proposed else C_BORDER,
            lw=1.6 if is_proposed else 0.8)
        offset_extra = sw*0.55 + 0.05
    else:
        offset_extra = 0
    text(x0 + sw + offset_extra + 0.18, leg_y + leg_h/2, tag,
         fs=10, weight='bold', ha='left')
    text(x0 + sw + offset_extra + 0.68, leg_y + leg_h/2, desc,
         fs=9, ha='left', color=C_SUB)

plt.savefig(os.path.join(OUT, 'fig_framework.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT, 'fig_framework.png'), bbox_inches='tight', dpi=300)
plt.close()
print('Saved:', os.path.join(OUT, 'fig_framework.pdf'))
