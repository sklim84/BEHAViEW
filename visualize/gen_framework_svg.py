"""
Generate BECON framework figure as SVG → PDF (horizontal layout).
Usage: python visualize/gen_framework_svg.py
"""
import cairosvg
import os

OUT_DIR = '_paper/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Colors
C_INPUT = '#E8EAF6'
C_VIEW1 = '#BBDEFB'
C_VIEW2 = '#FFE0B2'
C_ENCODER = '#C8E6C9'
C_POOL = '#E1BEE7'
C_LOSS = '#FFCDD2'
C_EVAL = '#FFF9C4'
C_BORDER = '#455A64'
C_TEXT = '#212121'
C_SUB = '#616161'
C_ARROW = '#546E7A'
C_AXIS = '#1565C0'
C_PROPOSED = '#FFAB91'

# ============ Grid System ============
# All main boxes: same width and height
BW = 150        # box width (columns 2-4)
BH = 90         # box height
GAP_X = 40      # horizontal gap between columns
GAP_Y = 60      # vertical gap between top/bottom rows

# Input box (smaller)
IW = 120
IH = 60

# Output boxes (smaller)
OW = 120
OH = 48

# Row positions (top of box)
ROW_T = 35                      # top row
ROW_B = ROW_T + BH + GAP_Y     # bottom row
ROW_MID = ROW_T + BH // 2      # center of top row (for reference)
CY_T = ROW_T + BH // 2         # center Y of top boxes
CY_B = ROW_B + BH // 2         # center Y of bottom boxes
CY_M = (CY_T + CY_B) // 2     # vertical midpoint

# Column positions (left edge of box)
COL1 = 20                           # Input
COL2 = COL1 + IW + GAP_X           # Views
COL3 = COL2 + BW + GAP_X           # Encoder
COL4 = COL3 + BW + GAP_X           # Pooling
COL5 = COL4 + BW + GAP_X           # Loss/Eval

W = COL5 + OW + 25                 # total width
H = ROW_B + BH + 55                # total height (+ legend)

# Centers
CX2 = COL2 + BW // 2
CX3 = COL3 + BW // 2
CX4 = COL4 + BW // 2
CX5 = COL5 + OW // 2
CX1 = COL1 + IW // 2


def box(x, y, w, h, fill, rx=8, sw=1.3):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{C_BORDER}" stroke-width="{sw}"/>'


def txt(x, y, content, size=11, weight='bold', color=C_TEXT, anchor='middle'):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Helvetica, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{content}</text>'


def arrow(x1, y1, x2, y2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C_ARROW}" stroke-width="1.6" marker-end="url(#ah)"/>'


def curve(x1, y1, cx, cy, x2, y2):
    return f'<path d="M {x1},{y1} Q {cx},{cy} {x2},{y2}" fill="none" stroke="{C_ARROW}" stroke-width="1.6" marker-end="url(#ah)"/>'


def elbow_h(x1, y1, x2, y2):
    """Horizontal-first elbow: go right to midpoint X, then turn to target."""
    mx = (x1 + x2) // 2
    return f'<path d="M {x1},{y1} L {mx},{y1} L {mx},{y2} L {x2},{y2}" fill="none" stroke="{C_ARROW}" stroke-width="1.6" marker-end="url(#ah)"/>'


LEGEND_Y = H - 45

svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <marker id="ah" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
    <polygon points="0 0, 7 2.5, 0 5" fill="{C_ARROW}"/>
  </marker>
</defs>
<rect width="{W}" height="{H}" fill="white"/>

<!-- ===== COL 1: Input ===== -->
{box(COL1, CY_M - IH//2, IW, IH, C_INPUT)}
{txt(CX1, CY_M - 8, 'Transaction', 11)}
{txt(CX1, CY_M + 7, 'Graph', 11)}
{txt(CX1, CY_M + 22, 'G = (V, E, X)', 8, 'normal', C_SUB)}

<!-- Elbow arrows: input → views -->
{elbow_h(COL1+IW, CY_M - 5, COL2, CY_T)}
{elbow_h(COL1+IW, CY_M + 5, COL2, CY_B)}

<!-- ===== COL 2: Views ===== -->
{box(COL2, ROW_T, BW, BH, C_VIEW1)}
{txt(CX2, ROW_T + 18, 'View 1', 12)}
{txt(CX2, ROW_T + 34, 'Augmented Graph', 9.5, 'normal', C_TEXT)}
{txt(CX2, ROW_T + 49, 'EdgeRemoving', 8.5, 'normal', C_SUB)}
{txt(CX2, ROW_T + 61, 'FeatureMasking', 8.5, 'normal', C_SUB)}
{txt(CX2, ROW_T + 79, 'S-S 15% : S-B 85%', 8.5, 'bold', '#C62828')}

{box(COL2, ROW_B, BW, BH, C_VIEW2)}
{txt(CX2, ROW_B + 18, 'View 2', 12)}
{txt(CX2, ROW_B + 34, 'Behavioral k-NN', 9.5, 'normal', C_TEXT)}
{txt(CX2, ROW_B + 49, 'k-NN(x_behav, k=10)', 8.5, 'normal', C_SUB)}
{txt(CX2, ROW_B + 61, 'cosine similarity', 8.5, 'normal', C_SUB)}
{txt(CX2, ROW_B + 79, 'S-S 41% : S-B 59%', 8.5, 'bold', '#2E7D32')}

<!-- Axis 1 label -->
{txt(CX2, CY_M + 4, 'View Construction ↕', 8.5, 'bold', C_AXIS)}

<!-- Arrows: views → encoder -->
{arrow(COL2 + BW, CY_T, COL3, CY_T)}
{arrow(COL2 + BW, CY_B, COL3, CY_B)}

<!-- ===== COL 3: Encoder ===== -->
{box(COL3, ROW_T, BW, BH, C_ENCODER)}
{txt(CX3, CY_T - 8, 'GNN Encoder', 11)}
{txt(CX3, CY_T + 8, 'GCNConv + BN', 8.5, 'normal', C_SUB)}
{txt(CX3, CY_T + 20, '+ PReLU + Dropout', 8.5, 'normal', C_SUB)}

{box(COL3, ROW_B, BW, BH, C_ENCODER)}
{txt(CX3, CY_B - 8, 'GNN Encoder', 11)}
{txt(CX3, CY_B + 8, 'GCNConv + BN', 8.5, 'normal', C_SUB)}
{txt(CX3, CY_B + 20, '+ PReLU + Dropout', 8.5, 'normal', C_SUB)}

<!-- Shared weights bracket (right side of encoder col) -->
<path d="M {COL3+BW+4},{ROW_T+8} L {COL3+BW+10},{ROW_T+8} L {COL3+BW+10},{CY_M-5} L {COL3+BW+16},{CY_M} L {COL3+BW+10},{CY_M+5} L {COL3+BW+10},{ROW_B+BH-8} L {COL3+BW+4},{ROW_B+BH-8}" fill="none" stroke="{C_AXIS}" stroke-width="1"/>
{txt(COL3+BW+22, CY_M-3, 'shared', 7.5, 'normal', C_AXIS, 'start')}
{txt(COL3+BW+22, CY_M+8, 'weights', 7.5, 'normal', C_AXIS, 'start')}

<!-- Arrows: encoder → pooling -->
{arrow(COL3 + BW, CY_T, COL4, CY_T)}
{arrow(COL3 + BW, CY_B, COL4, CY_B)}

<!-- ===== COL 4: Subgraph Pooling ===== -->
{box(COL4, ROW_T, BW, BH, C_POOL)}
{txt(CX4, CY_T - 10, 'Subgraph', 11)}
{txt(CX4, CY_T + 5, 'Pooling', 11)}
{txt(CX4, CY_T + 22, 'mean(h_i ∪ N(i))', 8.5, 'normal', C_SUB)}

{box(COL4, ROW_B, BW, BH, C_POOL)}
{txt(CX4, CY_B - 10, 'Subgraph', 11)}
{txt(CX4, CY_B + 5, 'Pooling', 11)}
{txt(CX4, CY_B + 22, 'mean(h_i ∪ N(i))', 8.5, 'normal', C_SUB)}

<!-- Axis 2 label -->
{txt(CX4, CY_M + 4, 'Contrastive Level ↕', 8.5, 'bold', C_AXIS)}

<!-- Elbow arrows: both pooling → Loss (converge) -->
{elbow_h(COL4+BW, CY_T, COL5, CY_M - 18)}
{elbow_h(COL4+BW, CY_B, COL5, CY_M + 2)}

<!-- ===== COL 5: Loss + Eval (vertically stacked, centered) ===== -->
{box(COL5, CY_M - 30, OW, OH, C_LOSS)}
{txt(CX5, CY_M - 12, 'Contrastive', 10)}
{txt(CX5, CY_M + 3, 'Loss', 10)}

<!-- Arrow: loss → eval -->
{arrow(CX5, CY_M + OH - 30, CX5, CY_M + OH - 14)}

{box(COL5, CY_M + OH - 12, OW, OH + 8, C_EVAL)}
{txt(CX5, CY_M + OH + 6, '[h₁ ∥ h₂]', 10)}
{txt(CX5, CY_M + OH + 21, '→ LogReg', 9, 'normal', C_SUB)}
{txt(CX5, CY_M + OH + 36, 'F1_susp', 8.5, 'normal', '#C62828')}

<!-- ===== Legend ===== -->
{box(COL2 - 30, LEGEND_Y, COL4 + BW - COL2 + 160, 34, '#FAFAFA', 6, 0.8)}

{box(COL2 - 15, LEGEND_Y + 8, 12, 16, C_VIEW1, 3, 0.8)}
{txt(COL2 + 5, LEGEND_Y + 20, '(a)', 9, 'bold', C_TEXT, 'start')}
{txt(COL2 + 28, LEGEND_Y + 20, 'Aug + Node', 8, 'normal', C_SUB, 'start')}

{box(CX2 + 30, LEGEND_Y + 8, 12, 16, C_VIEW2, 3, 0.8)}
{txt(CX2 + 50, LEGEND_Y + 20, '(b)', 9, 'bold', C_TEXT, 'start')}
{txt(CX2 + 73, LEGEND_Y + 20, 'Behav + Node', 8, 'normal', C_SUB, 'start')}

{box(CX3 + 25, LEGEND_Y + 8, 12, 16, C_POOL, 3, 0.8)}
{txt(CX3 + 45, LEGEND_Y + 20, '(c)', 9, 'bold', C_TEXT, 'start')}
{txt(CX3 + 68, LEGEND_Y + 20, 'Aug + Subgraph', 8, 'normal', C_SUB, 'start')}

{box(CX4 + 25, LEGEND_Y + 8, 12, 16, C_PROPOSED, 3, 1.3)}
{txt(CX4 + 45, LEGEND_Y + 20, '(d)', 9, 'bold', C_TEXT, 'start')}
{txt(CX4 + 68, LEGEND_Y + 20, 'Behav + Subgraph ★', 8, 'bold', C_TEXT, 'start')}

</svg>'''

# Save
svg_path = os.path.join(OUT_DIR, 'fig_framework.svg')
with open(svg_path, 'w', encoding='utf-8') as f:
    f.write(svg)
print(f'Saved: {svg_path}')

pdf_path = os.path.join(OUT_DIR, 'fig_framework.pdf')
cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to=pdf_path)
print(f'Saved: {pdf_path}')

png_path = os.path.join(OUT_DIR, 'fig_framework.png')
cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=png_path, scale=2)
print(f'Saved: {png_path}')
