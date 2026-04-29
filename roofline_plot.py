"""
Generate roofline.svg for the CS 259 mini-project (TITAN V, sm_70).

Stdlib-only (no matplotlib): emits a self-contained SVG that any browser, vector
editor, or report tool can render or convert to PNG (e.g. `rsvg-convert
roofline.svg -o roofline.png`).

Two ceilings: HBM2 bandwidth (slope) and Tensor Core peak (ceiling).
Four data points: theoretical and achieved AI for Conv1 B=16 and Conv2 B=16.

Achieved AI uses DRAM bytes from `ncu_metrics.csv`:
  achieved_AI = algorithmic_FLOPs / (dram__bytes_read.sum + dram__bytes_write.sum)
Achieved performance uses kernel duration from the same ncu run:
  achieved_GFLOPS = algorithmic_FLOPs / gpu__time_duration.sum
"""
import math

PEAK_TENSOR_TFLOPS = 110.0          # FP16 input, FP32 accumulate
PEAK_BW_GBS        = 652.8          # HBM2
RIDGE = (PEAK_TENSOR_TFLOPS * 1e3) / PEAK_BW_GBS  # ≈ 168 FLOP/byte

# (label, AI [FLOP/byte], achieved GFLOPS or None for theoretical, color, marker)
# Theoretical AI is identical for db and no_db (algorithm-only); achieved AI/GFLOPS
# differ. Achieved numbers from ncu_metrics.csv (db) and ncu_metrics_no_db.csv (no_db).
points = [
    ("Conv1 theoretical",   383.8,  None,    "#1f77b4", "circle"),
    ("Conv2 theoretical",  1031.0,  None,    "#ff7f0e", "circle"),
    ("Conv1 db",              92.9, 5123.0,  "#1f77b4", "square"),
    ("Conv2 db",              80.3, 1176.0,  "#ff7f0e", "square"),
    ("Conv1 no_db",          102.9, 5288.0,  "#1f77b4", "triangle"),
    ("Conv2 no_db",           80.4, 1233.0,  "#ff7f0e", "triangle"),
]

# canvas
W, H = 760, 560
ML, MR, MT, MB = 80, 30, 50, 60     # margins
PW, PH = W - ML - MR, H - MT - MB

# axes (log-log)
X_MIN, X_MAX = 0.1, 1e4              # AI in FLOP/byte
Y_MIN, Y_MAX = 10.0, 3e5             # GFLOPS

def x_to_px(x):
    return ML + (math.log10(x) - math.log10(X_MIN)) / (math.log10(X_MAX) - math.log10(X_MIN)) * PW

def y_to_px(y):
    return MT + PH - (math.log10(y) - math.log10(Y_MIN)) / (math.log10(Y_MAX) - math.log10(Y_MIN)) * PH

def ceiling(ai):
    return min(ai * PEAK_BW_GBS, PEAK_TENSOR_TFLOPS * 1e3)

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">')
svg.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# grid + tick labels
for decade in range(-1, 5):
    x = 10 ** decade
    if X_MIN <= x <= X_MAX:
        px = x_to_px(x)
        svg.append(f'<line x1="{px:.1f}" y1="{MT}" x2="{px:.1f}" y2="{MT+PH}" '
                   f'stroke="#ddd" stroke-width="0.5"/>')
        svg.append(f'<text x="{px:.1f}" y="{MT+PH+18}" text-anchor="middle" '
                   f'fill="#444">10<tspan font-size="9" baseline-shift="super">{decade}</tspan></text>')
for decade in range(1, 6):
    y = 10 ** decade
    if Y_MIN <= y <= Y_MAX:
        py = y_to_px(y)
        svg.append(f'<line x1="{ML}" y1="{py:.1f}" x2="{ML+PW}" y2="{py:.1f}" '
                   f'stroke="#ddd" stroke-width="0.5"/>')
        svg.append(f'<text x="{ML-8}" y="{py+4:.1f}" text-anchor="end" '
                   f'fill="#444">10<tspan font-size="9" baseline-shift="super">{decade}</tspan></text>')

# axes
svg.append(f'<line x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" stroke="black" stroke-width="1"/>')
svg.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="black" stroke-width="1"/>')
svg.append(f'<text x="{ML+PW/2}" y="{H-15}" text-anchor="middle" font-size="13">'
           f'Arithmetic Intensity (FLOP/byte)</text>')
svg.append(f'<text transform="rotate(-90 18 {MT+PH/2})" x="18" y="{MT+PH/2}" '
           f'text-anchor="middle" font-size="13">Performance (GFLOPS)</text>')
svg.append(f'<text x="{ML+PW/2}" y="{MT-20}" text-anchor="middle" font-size="14" '
           f'font-weight="bold">Roofline — TITAN V (sm_70), conv.cu vs conv_no_db.cu B=16</text>')

# ridge marker (vertical dotted line)
ridge_px = x_to_px(RIDGE)
svg.append(f'<line x1="{ridge_px:.1f}" y1="{MT}" x2="{ridge_px:.1f}" y2="{MT+PH}" '
           f'stroke="#888" stroke-dasharray="3,3" stroke-width="0.7"/>')
svg.append(f'<text x="{ridge_px+5:.1f}" y="{MT+15}" font-size="10" fill="#666">'
           f'ridge ≈ {RIDGE:.0f} FLOP/byte</text>')

# roofline (piecewise: BW slope, then tensor ceiling)
# build polyline samples in log space
xs = []
n = 400
for i in range(n + 1):
    xs.append(X_MIN * (X_MAX / X_MIN) ** (i / n))
ys = [ceiling(x) for x in xs]
pts = " ".join(f"{x_to_px(x):.1f},{y_to_px(y):.1f}" for x, y in zip(xs, ys))
svg.append(f'<polyline points="{pts}" stroke="black" stroke-width="2" fill="none"/>')

# extend dashed parts past the corner for visual cue
# BW slope past ridge
ai_past = X_MAX
y_past_bw = ai_past * PEAK_BW_GBS
if y_past_bw > Y_MAX:
    y_past_bw = Y_MAX
svg.append(f'<line x1="{x_to_px(RIDGE):.1f}" y1="{y_to_px(PEAK_TENSOR_TFLOPS*1e3):.1f}" '
           f'x2="{x_to_px(min(ai_past, Y_MAX/PEAK_BW_GBS)):.1f}" '
           f'y2="{y_to_px(min(ai_past*PEAK_BW_GBS, Y_MAX)):.1f}" '
           f'stroke="black" stroke-dasharray="4,4" stroke-width="0.8" opacity="0.45"/>')
# tensor ceiling left of ridge
svg.append(f'<line x1="{x_to_px(X_MIN):.1f}" y1="{y_to_px(PEAK_TENSOR_TFLOPS*1e3):.1f}" '
           f'x2="{x_to_px(RIDGE):.1f}" y2="{y_to_px(PEAK_TENSOR_TFLOPS*1e3):.1f}" '
           f'stroke="black" stroke-dasharray="4,4" stroke-width="0.8" opacity="0.45"/>')

# label the ceilings
svg.append(f'<text x="{x_to_px(2000):.1f}" y="{y_to_px(PEAK_TENSOR_TFLOPS*1e3)-8:.1f}" '
           f'text-anchor="middle" font-size="11" fill="#333">'
           f'Tensor peak {PEAK_TENSOR_TFLOPS:.0f} TFLOPS</text>')
svg.append(f'<text x="{x_to_px(0.5):.1f}" y="{y_to_px(0.5*PEAK_BW_GBS)-8:.1f}" '
           f'font-size="11" fill="#333" '
           f'transform="rotate(-30 {x_to_px(0.5):.1f} {y_to_px(0.5*PEAK_BW_GBS):.1f})">'
           f'HBM2 {PEAK_BW_GBS:.0f} GB/s</text>')

# data points + labels
def draw_marker(cx, cy, color, kind, filled):
    fill = color if filled else "white"
    if kind == "circle":
        return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{fill}" '
                f'stroke="{color}" stroke-width="1.6"/>')
    elif kind == "triangle":
        # equilateral, point up; matches r=6 visual size
        pts = f"{cx:.1f},{cy-7:.1f} {cx-7:.1f},{cy+5:.1f} {cx+7:.1f},{cy+5:.1f}"
        return (f'<polygon points="{pts}" fill="{fill}" '
                f'stroke="{color}" stroke-width="1.6"/>')
    else:  # square
        return (f'<rect x="{cx-6:.1f}" y="{cy-6:.1f}" width="12" height="12" '
                f'fill="{fill}" stroke="{color}" stroke-width="1.6"/>')

# Per-point dx/dy offset for the text label, in px from marker center.
# db (square) labels go to lower-left so they don't collide with the
# nearby no_db (triangle) labels which go to upper-right.
label_offsets = [
    (8, -10),    # Conv1 theoretical
    (8, -10),    # Conv2 theoretical
    (-72, 18),   # Conv1 db
    (-72, 18),   # Conv2 db
    (10, -10),   # Conv1 no_db
    (10, -10),   # Conv2 no_db
]
for (label, ai, gflops, color, kind), (dx, dy) in zip(points, label_offsets):
    if gflops is None:
        # theoretical: place on the ceiling
        y_plot = ceiling(ai)
        filled = False
    else:
        y_plot = gflops
        filled = True
    cx = x_to_px(ai)
    cy = y_to_px(y_plot)
    svg.append(draw_marker(cx, cy, color, kind, filled))
    svg.append(f'<text x="{cx+dx:.1f}" y="{cy+dy:.1f}" font-size="10" fill="{color}">{label}</text>')

# legend
lx, ly = ML + 15, MT + PH - 110
svg.append(f'<rect x="{lx-8}" y="{ly-15}" width="240" height="112" fill="white" '
           f'stroke="#bbb" stroke-width="0.5" opacity="0.92"/>')
legend_rows = [
    ("circle",   False, "theoretical (on tensor ceiling)"),
    ("square",   True,  "conv.cu — double-buffered (db)"),
    ("triangle", True,  "conv_no_db.cu — single buffer"),
]
for i, (kind, filled, text) in enumerate(legend_rows):
    yy = ly + i * 16
    svg.append(draw_marker(lx, yy, "#444", kind, filled))
    svg.append(f'<text x="{lx+14}" y="{yy+4}" font-size="11" fill="#222">{text}</text>')
# layer color swatches
swatch_y = ly + 56
svg.append(f'<rect x="{lx-3}" y="{swatch_y-7}" width="10" height="10" fill="#1f77b4"/>')
svg.append(f'<text x="{lx+12}" y="{swatch_y+3}" font-size="11" fill="#222">Conv1 B=16 (224×224, 64→64)</text>')
svg.append(f'<rect x="{lx-3}" y="{swatch_y+10}" width="10" height="10" fill="#ff7f0e"/>')
svg.append(f'<text x="{lx+12}" y="{swatch_y+20}" font-size="11" fill="#222">Conv2 B=16 (14×14, 512→512)</text>')

svg.append('</svg>')

with open("roofline.svg", "w") as f:
    f.write("\n".join(svg))
print("wrote roofline.svg")
