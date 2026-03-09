"""
VIZ 2 — Dual Heatmap  (Suggested Alternative)
===============================================
Layout  : One heatmap panel per Supplier, displayed side-by-side
Rows    : Cheese Types
Columns : Tests
Cell    : Mean test value (z-score normalised per test so all tests are comparable)
Colour  : Diverging — blue (low) → white → red (high) within each test column
Bonus   : Delta heatmap (Supplier A minus Supplier B) as a 3rd panel
          showing where the two suppliers differ most

Why it's useful for non-tech audience
──────────────────────────────────────
• No filters — entire dataset visible at once
• Spot which cheese + test combinations are outliers at a glance
• Delta panel immediately highlights quality/consistency gaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    exc  = {SUPPLIER_COL, CHEESE_COL}
    TEST_COLS = [c for c in df.select_dtypes(include=np.number).columns if c not in exc][:5]
    print(f"  Test cols    : {TEST_COLS}\n")

for c in TEST_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[SUPPLIER_COL, CHEESE_COL] + TEST_COLS)
df[SUPPLIER_COL] = df[SUPPLIER_COL].astype(str)
df[CHEESE_COL]   = df[CHEESE_COL].astype(str)

# ── ③ BUILD PIVOT TABLES ──────────────────────────────────────────────────────
supplier_order = sorted(df[SUPPLIER_COL].unique())
cheese_order   = sorted(df[CHEESE_COL].unique())

pivots = {}
for sup in supplier_order:
    sub = df[df[SUPPLIER_COL] == sup]
    piv = (sub.groupby(CHEESE_COL)[TEST_COLS]
              .mean()
              .reindex(cheese_order))
    pivots[sup] = piv

# z-score normalise each test column across all data so colour scale is comparable
all_means = pd.concat(pivots.values())
z_pivots  = {}
for sup in supplier_order:
    z  = pivots[sup].copy()
    for col in TEST_COLS:
        mu = all_means[col].mean()
        sd = all_means[col].std()
        z[col] = (z[col] - mu) / (sd + 1e-9)
    z_pivots[sup] = z

# delta panel (Sup A − Sup B), only when exactly 2 suppliers
show_delta = len(supplier_order) == 2
if show_delta:
    delta = z_pivots[supplier_order[0]] - z_pivots[supplier_order[1]]

# ── ④ DRAW ────────────────────────────────────────────────────────────────────
BG   = "#0F0F1A"
n_sup = len(supplier_order)
n_panels = n_sup + (1 if show_delta else 0)

n_cheese = len(cheese_order)
n_tests  = len(TEST_COLS)

cell_w = max(1.2, 18 / n_tests)
cell_h = max(0.36, 18 / n_cheese)
fig_w  = n_panels * (n_tests * cell_w + 2.5) + 1.5
fig_h  = n_cheese * cell_h + 3.5

fig, axes = plt.subplots(1, n_panels,
                         figsize=(fig_w, fig_h),
                         facecolor=BG,
                         gridspec_kw={"wspace": 0.12})
if n_panels == 1:
    axes = [axes]

# shared colour map limits
vmax_sup = max(abs(z_pivots[s]).values.max() for s in supplier_order)
norm_sup = mcolors.TwoSlopeNorm(vmin=-vmax_sup, vcenter=0, vmax=vmax_sup)
cmap_sup = plt.cm.RdYlBu_r   # red = high, blue = low

SUPPLIER_ACCENT = ["#E74C3C","#2471A3","#27AE60","#D35400"]

for pidx, sup in enumerate(supplier_order):
    ax = axes[pidx]
    ax.set_facecolor(BG)

    data_z = z_pivots[sup].values
    im = ax.imshow(data_z, aspect="auto",
                   cmap=cmap_sup, norm=norm_sup, zorder=2)

    # grid lines
    for x in np.arange(-0.5, n_tests, 1):
        ax.axvline(x, color=BG, lw=1.2, zorder=3)
    for y in np.arange(-0.5, n_cheese, 1):
        ax.axhline(y, color=BG, lw=0.8, zorder=3)

    # annotate cells with raw mean (not z-score)
    raw = pivots[sup].values
    for r in range(n_cheese):
        for c_ in range(n_tests):
            val = raw[r, c_]
            if not np.isnan(val):
                z_v = data_z[r, c_]
                txt_color = "white" if abs(z_v) > 0.8 else "#222222"
                ax.text(c_, r, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=max(5.5, min(8, 90 / n_cheese)),
                        color=txt_color, zorder=4)

    # axis labels
    ax.set_xticks(range(n_tests))
    ax.set_xticklabels(TEST_COLS, rotation=30, ha="right",
                       fontsize=9, color="#CCCCDD")
    ax.set_yticks(range(n_cheese))
    if pidx == 0:
        yl = [str(c)[:22] + "…" if len(str(c)) > 22 else str(c)
              for c in cheese_order]
        ax.set_yticklabels(yl, fontsize=max(6, min(9, 200 / n_cheese)),
                           color="#CCCCDD")
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)

    accent = SUPPLIER_ACCENT[pidx % len(SUPPLIER_ACCENT)]
    ax.set_title(f"Supplier: {sup}", color=accent,
                 fontsize=13, fontweight="bold", pad=10)

    # border highlight
    for sp in ax.spines.values():
        sp.set_edgecolor(accent)
        sp.set_linewidth(2)

    # colorbar per panel
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.08)
    cb  = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, colors="#AAAACC")
    cb.set_label("z-score", color="#AAAACC", fontsize=8)
    cax.yaxis.set_tick_params(color="#AAAACC")
    plt.setp(cb.ax.yaxis.get_ticklines(), color="#AAAACC")

# ── delta panel ───────────────────────────────────────────────────────────────
if show_delta:
    ax = axes[-1]
    ax.set_facecolor(BG)
    dmax = np.nanmax(np.abs(delta.values))
    norm_d = mcolors.TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)
    im_d = ax.imshow(delta.values, aspect="auto",
                     cmap="PiYG", norm=norm_d, zorder=2)

    for x in np.arange(-0.5, n_tests, 1):
        ax.axvline(x, color=BG, lw=1.2, zorder=3)
    for y in np.arange(-0.5, n_cheese, 1):
        ax.axhline(y, color=BG, lw=0.8, zorder=3)

    for r in range(n_cheese):
        for c_ in range(n_tests):
            v = delta.values[r, c_]
            if not np.isnan(v):
                tc = "white" if abs(v) > dmax * 0.5 else "#222222"
                ax.text(c_, r, f"{v:+.2f}",
                        ha="center", va="center",
                        fontsize=max(5.5, min(7.5, 90 / n_cheese)),
                        color=tc, zorder=4)

    ax.set_xticks(range(n_tests))
    ax.set_xticklabels(TEST_COLS, rotation=30, ha="right",
                       fontsize=9, color="#CCCCDD")
    ax.set_yticks(range(n_cheese))
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.set_title(f"Δ  {supplier_order[0]}  minus  {supplier_order[1]}",
                 color="#A9DFBF", fontsize=12, fontweight="bold", pad=10)
    for sp in ax.spines.values():
        sp.set_edgecolor("#A9DFBF")
        sp.set_linewidth(2)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.08)
    cb  = plt.colorbar(im_d, cax=cax)
    cb.ax.tick_params(labelsize=7, colors="#AAAACC")
    cb.set_label("Δ z-score", color="#AAAACC", fontsize=8)
    plt.setp(cb.ax.yaxis.get_ticklines(), color="#AAAACC")

# ── titles ────────────────────────────────────────────────────────────────────
fig.suptitle("Cheese COA — Mean Test Values per Supplier  (z-score normalised per test)",
             fontsize=14, fontweight="bold", color="white", y=1.01)
fig.text(0.5, -0.01,
         "Cell value = raw mean  |  Cell colour = z-score within each test column  |  "
         "Δ panel = positive (green) means Supplier A is higher",
         ha="center", fontsize=8.5, color="#888899")

plt.savefig("viz2_heatmap.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("✓ Saved → viz2_heatmap.png")
plt.close()
