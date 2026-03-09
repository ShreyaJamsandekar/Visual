"""
VIZ 3 — Violin + Box Overlay  (Distribution Deep-Dive)
========================================================
Layout  : One row per Test (5 rows)
X-axis  : Cheese Types
Violins : Split violin — left half = Supplier A, right half = Supplier B
          so both distributions share the same x-slot (no doubling of columns)
Inside  : Thin box-plot whiskers overlaid for median / IQR reference
Why     : Shows the full shape of distributions (not just dots or means)
          Great for spotting bimodal batches, outliers, wide variance

Handles 20 K+ points by aggregating into kernel density — chart stays clean
regardless of sample size.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

# ── ① CONFIGURATION ───────────────────────────────────────────────────────────
FILE_PATH    = "COA data.xlsx"
SUPPLIER_COL = None
CHEESE_COL   = None
TEST_COLS    = None

# ── ② LOAD & DETECT ──────────────────────────────────────────────────────────
df = pd.read_excel(FILE_PATH)
print(f"\n✓ Loaded  {len(df):,} rows")

def first_match(df, kws):
    for kw in kws:
        for col in df.columns:
            if kw.lower() in str(col).lower():
                return col
    return None

if SUPPLIER_COL is None:
    SUPPLIER_COL = first_match(df, ["supplier","vendor","company","mfr"])
    print(f"  Supplier col : '{SUPPLIER_COL}'")
if CHEESE_COL is None:
    CHEESE_COL = first_match(df, ["cheese","product","type","variety","item","name"])
    print(f"  Cheese col   : '{CHEESE_COL}'")
if TEST_COLS is None:
    exc = {SUPPLIER_COL, CHEESE_COL}
    TEST_COLS = [c for c in df.select_dtypes(include=np.number).columns if c not in exc][:5]
    print(f"  Test cols    : {TEST_COLS}\n")

for c in TEST_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[SUPPLIER_COL, CHEESE_COL] + TEST_COLS)
df[SUPPLIER_COL] = df[SUPPLIER_COL].astype(str)
df[CHEESE_COL]   = df[CHEESE_COL].astype(str)

# ── ③ PALETTE ─────────────────────────────────────────────────────────────────
supplier_order = sorted(df[SUPPLIER_COL].unique())
cheese_order   = sorted(df[CHEESE_COL].unique())
n_cheese = len(cheese_order)

# Supplier colours — vivid + translucent fills
SUP_COLORS_FILL  = ["#E74C3C","#2471A3","#27AE60","#D35400","#8E44AD"]
SUP_COLORS_EDGE  = ["#FF8C7A","#5DADE2","#58D68D","#F0A030","#C39BD3"]

CHEESE_BG = [
    "#D6EAF820","#D5F5E320","#FDEDEC20","#FEF9E720","#F4ECF720",
    "#E8F8F520","#FDF2E920","#EBF5FB20","#E9F7EF20","#FDFEFE20",
    "#F9EBEA20","#EAF2FF20","#E8DAEF20","#D1F2EB20","#FCF3CF20",
    "#FADBD820","#D5D8DC20","#D2B4DE20","#A9DFBF20","#FAD7A020",
    "#A3E4D720","#AED6F120","#F9E79F20","#F5CBA720","#D7BDE220",
    "#A9CCE320","#A2D9CE20","#F8C47120","#82E0AA20","#F1948A20",
]
# opaque version for legend swatches
CHEESE_BG_LEGEND = [c.replace("20","FF") for c in CHEESE_BG]

BG = "#0E0E1C"

# ── ④ KDE VIOLIN HELPER ───────────────────────────────────────────────────────
def draw_half_violin(ax, x_center, values, side, color, edge_color,
                     width=0.40, n_pts=200, alpha_fill=0.55):
    """Draw one half of a split violin (left or right of x_center)."""
    if len(values) < 4:
        return
    values = values[~np.isnan(values)]
    lo, hi = np.percentile(values, [1, 99])
    ys = np.linspace(lo - (hi - lo) * 0.05,
                     hi + (hi - lo) * 0.05, n_pts)
    try:
        kde = gaussian_kde(values, bw_method="scott")
        ks  = kde(ys)
    except Exception:
        return

    ks_norm = ks / ks.max() * width   # scale to desired half-width

    if side == "left":
        xs_fill = x_center - ks_norm
        xs_edge = list(x_center - ks_norm) + [x_center, x_center]
        ys_edge = list(ys) + [ys[-1], ys[0]]
        ax.fill_betweenx(ys, x_center, xs_fill,
                         color=color, alpha=alpha_fill, zorder=3)
    else:
        xs_fill = x_center + ks_norm
        xs_edge = [x_center, x_center] + list(x_center + ks_norm)
        ys_edge = [ys[0], ys[-1]] + list(ys)
        ax.fill_betweenx(ys, x_center, xs_fill,
                         color=color, alpha=alpha_fill, zorder=3)

    ax.plot(xs_edge, ys_edge, color=edge_color, lw=1.0, alpha=0.8, zorder=4)

    # IQR box
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    bw = width * 0.18
    sign = -1 if side == "left" else 1
    ax.plot([x_center, x_center + sign * bw * 2],
            [med, med], color="white", lw=1.8, zorder=5)
    ax.plot([x_center, x_center + sign * bw],
            [q1, q1], color=edge_color, lw=1.0, zorder=5)
    ax.plot([x_center, x_center + sign * bw],
            [q3, q3], color=edge_color, lw=1.0, zorder=5)
    ax.vlines(x_center + sign * bw, q1, q3,
              colors=edge_color, lw=0.8, zorder=5)

# ── ⑤ DRAW ────────────────────────────────────────────────────────────────────
n_tests  = len(TEST_COLS)
fig_w    = max(24, n_cheese * 0.72)
fig, axes = plt.subplots(n_tests, 1,
                         figsize=(fig_w, 5.5 * n_tests + 1.5),
                         facecolor=BG)
if n_tests == 1:
    axes = [axes]

fig.suptitle("Cheese COA — Test Value Distributions by Supplier & Cheese Type",
             fontsize=16, fontweight="bold", color="white", y=1.002)

sides = ["left", "right"] + ["left"] * 8   # first 2 suppliers get split violin

for tidx, (test_col, ax) in enumerate(zip(TEST_COLS, axes)):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2A2A3C")

    # background bands
    for i, cheese in enumerate(cheese_order):
        bg_hex = CHEESE_BG[i % len(CHEESE_BG)]
        # convert 8-char hex with alpha to rgba
        r, g, b = int(bg_hex[1:3], 16), int(bg_hex[3:5], 16), int(bg_hex[5:7], 16)
        ax.axvspan(i - 0.5, i + 0.5,
                   facecolor=(r/255, g/255, b/255, 0.28), zorder=0)
        if i > 0:
            ax.axvline(i - 0.5, color="#1E1E2E", lw=0.7, zorder=1)

    # draw violins
    for i, cheese in enumerate(cheese_order):
        cheese_df = df[df[CHEESE_COL] == cheese]
        for sidx, sup in enumerate(supplier_order):
            vals = cheese_df[cheese_df[SUPPLIER_COL] == sup][test_col].dropna().values
            if len(vals) < 3:
                continue
            side = sides[sidx] if sidx < 2 else ("left" if sidx % 2 == 0 else "right")
            fc   = SUP_COLORS_FILL[sidx % len(SUP_COLORS_FILL)]
            ec   = SUP_COLORS_EDGE[sidx % len(SUP_COLORS_EDGE)]
            draw_half_violin(ax, i, vals, side, fc, ec,
                             width=0.42, alpha_fill=0.52)

    # axes
    ax.set_xticks(range(n_cheese))
    short = [str(c)[:16] + "…" if len(str(c)) > 16 else str(c)
             for c in cheese_order]
    ax.set_xticklabels(short, rotation=42, ha="right",
                       fontsize=7.8, color="#BBBBCC")
    ax.set_ylabel(test_col, color="white", fontsize=11, labelpad=8)
    ax.tick_params(axis="y", colors="#AAAACC", labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#1E1E2E", lw=0.6, zorder=0)
    ax.set_title(f" Test: {test_col}", loc="left",
                 color="#88CCFF", fontsize=12, pad=5)
    ax.set_xlim(-0.5, n_cheese - 0.5)

    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.04
    ax.set_ylim(ymin - pad, ymax + pad)

# ── ⑥ LEGENDS ────────────────────────────────────────────────────────────────
leg_kw = dict(facecolor="#15152A", edgecolor="#33334A",
              labelcolor="white", fontsize=9, framealpha=0.95)

sup_handles = []
for sidx, sup in enumerate(supplier_order):
    side_label = "◐ left half" if sidx == 0 else "◑ right half" if sidx == 1 else ""
    sup_handles.append(
        mpatches.Patch(facecolor=SUP_COLORS_FILL[sidx % len(SUP_COLORS_FILL)],
                       edgecolor=SUP_COLORS_EDGE[sidx % len(SUP_COLORS_EDGE)],
                       label=f"{sup}  {side_label}", linewidth=1.2, alpha=0.8)
    )

show_n  = min(n_cheese, 10)
bg_hdls = []
for i, c in enumerate(cheese_order[:show_n]):
    bg_hex = CHEESE_BG_LEGEND[i % len(CHEESE_BG_LEGEND)]
    r, g, b = int(bg_hex[1:3], 16), int(bg_hex[3:5], 16), int(bg_hex[5:7], 16)
    bg_hdls.append(
        mpatches.Patch(facecolor=(r/255, g/255, b/255, 0.55), label=str(c),
                       edgecolor="#555566", linewidth=0.7)
    )
if n_cheese > show_n:
    bg_hdls.append(mpatches.Patch(facecolor="none",
                                   label=f"… +{n_cheese - show_n} more",
                                   edgecolor="none", alpha=0))

l1 = axes[0].legend(handles=sup_handles, title="▌ Supplier (violin half)",
                    title_fontsize=10,
                    loc="upper right", bbox_to_anchor=(1.0, 1.40),
                    **leg_kw)
l1.get_title().set_color("#88CCFF")
l2 = axes[0].legend(handles=bg_hdls,
                    title="▌ Cheese Type (background)",
                    title_fontsize=9,
                    loc="upper left", bbox_to_anchor=(0.0, 1.40),
                    ncol=5, **leg_kw)
l2.get_title().set_color("#88CCFF")
axes[0].add_artist(l1)

fig.text(0.5, -0.006,
         "Each violin half shows the full distribution shape  |  White bar = median  |  Coloured bar = IQR  |  Wider violin = more data points at that value",
         ha="center", fontsize=8, color="#777788")

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("viz3_violin.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("✓ Saved → viz3_violin.png")
plt.close()
