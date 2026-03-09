"""
VIZ 3 — Split Violin  |  One chart per Test ID
================================================
• White background
• One PNG per Test ID
• X-axis  : Cheese Types — each column tinted with a unique background
• Y-axis  : Test result distribution
• Violin  : Left half = Supplier A, Right half = Supplier B
            (if >2 suppliers: side-by-side violins, spaced slightly)
• Inside  : White median bar + IQR bracket
• Legend  : RIGHT of chart — never overlaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde
import warnings, os
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  ① CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
FILE_PATH    = "COA data.xlsx"
SUPPLIER_COL = None
CHEESE_COL   = None
TEST_ID_COL  = None
VALUE_COL    = None
OUTPUT_DIR   = "."

# ──────────────────────────────────────────────────────────────────────────────
#  ② PALETTES
# ──────────────────────────────────────────────────────────────────────────────
CHEESE_BG = [
    "#D6EAF8","#D5F5E3","#FDEDEC","#FEF9E7","#F4ECF7",
    "#E8F8F5","#FDF2E9","#E8DAEF","#D1F2EB","#FCF3CF",
    "#FADBD8","#EBF5FB","#A9DFBF","#FAD7A0","#A3E4D7",
    "#AED6F1","#F9E79F","#F5CBA7","#D7BDE2","#A9CCE3",
    "#A2D9CE","#F8C471","#82E0AA","#F1948A","#85C1E9",
    "#76D7C4","#F0B27A","#BB8FCE","#73C6B6","#F7DC6F",
]
SUPPLIER_FILL = ["#C0392B","#1A5276","#1E8449","#784212","#6C3483","#0E6655"]
SUPPLIER_EDGE = ["#E74C3C","#2980B9","#27AE60","#CA6F1E","#8E44AD","#17A589"]

# ──────────────────────────────────────────────────────────────────────────────
#  ③ HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def find_col(df, kws):
    for kw in kws:
        for col in df.columns:
            if kw.lower() in str(col).lower():
                return col
    return None

def trunc(text, n=13):
    s = str(text)
    return (s[:n] + "…") if len(s) > n else s

def half_violin(ax, x, values, side, fill, edge, half_width=0.38, alpha=0.60):
    """Draw one half (left or right) of a violin at position x."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 4:
        return
    lo, hi = np.nanpercentile(vals, [1, 99])
    span    = hi - lo
    ys      = np.linspace(lo - span * 0.06, hi + span * 0.06, 250)
    try:
        ks  = gaussian_kde(vals, bw_method="scott")(ys)
    except Exception:
        return
    ks_norm = ks / ks.max() * half_width

    sign = -1 if side == "left" else 1
    xs_outer = x + sign * ks_norm

    # fill
    ax.fill_betweenx(ys, x, xs_outer,
                     color=fill, alpha=alpha, zorder=3, linewidth=0)
    # outline
    outline_x = ([x] + list(xs_outer) + [x]) if side == "right" \
                else ([x] + list(xs_outer) + [x])
    outline_y = ([ys[0]] + list(ys) + [ys[-1]])
    ax.plot(outline_x, outline_y, color=edge, lw=0.9, alpha=0.85, zorder=4)

    # statistics
    q1, med, q3 = np.nanpercentile(vals, [25, 50, 75])
    bw = half_width * 0.22
    # median
    ax.hlines(med, x, x + sign * bw * 2.2,
              colors="white", lw=2.2, zorder=5)
    # IQR bracket
    ax.vlines(x + sign * bw, q1, q3,
              colors=edge, lw=1.1, alpha=0.9, zorder=5)

# ──────────────────────────────────────────────────────────────────────────────
#  ④ LOAD
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_excel(FILE_PATH)
print(f"\n✓ Loaded  {len(df):,} rows")

if SUPPLIER_COL is None:
    SUPPLIER_COL = find_col(df, ["supplier","vendor","mfr","manufacturer","company"])
if CHEESE_COL is None:
    CHEESE_COL   = find_col(df, ["cheese","product","variety","item","sku"])
if TEST_ID_COL is None:
    TEST_ID_COL  = find_col(df, ["test id","testid","test_id","test name","test","parameter","analyte"])
if VALUE_COL is None:
    VALUE_COL    = find_col(df, ["result","value","reading","measurement","actual","amount"])
    if VALUE_COL is None:
        exc  = {SUPPLIER_COL, CHEESE_COL, TEST_ID_COL}
        nums = [c for c in df.select_dtypes(include=np.number).columns if c not in exc]
        VALUE_COL = nums[0] if nums else None

assert all([SUPPLIER_COL, CHEESE_COL, TEST_ID_COL, VALUE_COL]), \
    "Column detection failed — set column names manually."

df[VALUE_COL]    = pd.to_numeric(df[VALUE_COL], errors="coerce")
df[SUPPLIER_COL] = df[SUPPLIER_COL].astype(str).str.strip()
df[CHEESE_COL]   = df[CHEESE_COL].astype(str).str.strip()
df[TEST_ID_COL]  = df[TEST_ID_COL].astype(str).str.strip()
df = df.dropna(subset=[VALUE_COL])

cheese_order   = sorted(df[CHEESE_COL].unique())
supplier_order = sorted(df[SUPPLIER_COL].unique())
test_ids       = sorted(df[TEST_ID_COL].unique())
n_cheese       = len(cheese_order)
n_sup          = len(supplier_order)

print(f"  Suppliers : {n_sup}  |  Cheese : {n_cheese}  |  Tests : {len(test_ids)}\n")

cheese_bg_map = {c: CHEESE_BG[i % len(CHEESE_BG)] for i, c in enumerate(cheese_order)}

# Decide violin positions for ≥2 suppliers
# 2 suppliers → clean split (left / right)
# >2 suppliers → evenly spaced offsets within ±0.40
def sup_positions(n):
    if n == 1:  return [0.0], [0.38], ["right"]
    if n == 2:  return [-0.02, 0.02], [0.36, 0.36], ["left", "right"]
    offsets = np.linspace(-0.35, 0.35, n)
    return list(offsets), [0.30 / n * 1.5] * n, ["right"] * n

# ──────────────────────────────────────────────────────────────────────────────
#  ⑤ DRAW
# ──────────────────────────────────────────────────────────────────────────────
for test_id in test_ids:
    sub = df[df[TEST_ID_COL] == test_id]
    if sub.empty:
        continue

    fig_w = max(16, n_cheese * 0.78) + 2.8
    fig_h = 6.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.07, right=0.78, top=0.88, bottom=0.20)

    # ── cheese column backgrounds ----------------------------------------
    for i, cheese in enumerate(cheese_order):
        ax.axvspan(i - 0.5, i + 0.5,
                   facecolor=cheese_bg_map[cheese], alpha=1.0, zorder=0)
        if i > 0:
            ax.axvline(i - 0.5, color="white", lw=1.5, zorder=1)

    offsets, widths, default_sides = sup_positions(n_sup)

    for sidx, sup in enumerate(supplier_order):
        s_df   = sub[sub[SUPPLIER_COL] == sup]
        fill_c = SUPPLIER_FILL[sidx % len(SUPPLIER_FILL)]
        edge_c = SUPPLIER_EDGE[sidx % len(SUPPLIER_EDGE)]
        offset = offsets[sidx]
        hw     = widths[sidx]
        side   = default_sides[sidx]

        for i, cheese in enumerate(cheese_order):
            vals = s_df[s_df[CHEESE_COL] == cheese][VALUE_COL].dropna().values
            half_violin(ax, i + offset, vals, side,
                        fill_c, edge_c, half_width=hw)

    # ── x-axis -----------------------------------------------------------
    ax.set_xticks(range(n_cheese))
    ax.set_xticklabels([trunc(c, 13) for c in cheese_order],
                       rotation=40, ha="right", fontsize=8.5, color="#333333")
    ax.set_xlim(-0.5, n_cheese - 0.5)
    ax.tick_params(axis="x", length=0, pad=3)

    # ── y-axis -----------------------------------------------------------
    ax.set_ylabel("Result", fontsize=10, color="#444444", labelpad=6)
    ax.tick_params(axis="y", labelsize=9, colors="#555555")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", color="#E8E8E8", lw=0.6, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    ax.set_title(f"  {test_id}", fontsize=13, fontweight="bold",
                 color="#111111", pad=10, loc="left")

    # ── legend RIGHT of axes --------------------------------------------
    handles = [
        mpatches.Patch(facecolor=SUPPLIER_FILL[i % len(SUPPLIER_FILL)],
                       edgecolor=SUPPLIER_EDGE[i % len(SUPPLIER_EDGE)],
                       linewidth=1, label=s, alpha=0.85)
        for i, s in enumerate(supplier_order)
    ]
    ax.legend(
        handles=handles,
        title="Supplier",
        title_fontsize=9,
        fontsize=8.5,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        framealpha=1.0,
        edgecolor="#DDDDDD",
        facecolor="white",
    )

    # ── caption ----------------------------------------------------------
    note_parts = ["Violin width = data density", "White bar = median", "Bracket = IQR"]
    if n_sup == 2:
        note_parts.insert(0, "Left half = " + supplier_order[0]
                          + "  |  Right half = " + supplier_order[1])
    fig.text(0.01, -0.02, "  |  ".join(note_parts),
             fontsize=7.5, color="#888888", ha="left")

    # ── save -------------------------------------------------------------
    safe = str(test_id).replace("/","-").replace(" ","_").replace(":","-")
    out  = os.path.join(OUTPUT_DIR, f"viz3_{safe}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  ✓ Saved → {out}")
    plt.close()

print("\nDone — one violin chart per Test ID saved.")
