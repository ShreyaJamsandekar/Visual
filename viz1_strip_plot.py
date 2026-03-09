"""
VIZ 1 — Strip Plot  (Your Original Idea)
==========================================
Layout  : One subplot per test (5 rows total)
X-axis  : Cheese Types — each gets its own pastel background band
Y-axis  : Test value
Dots    : Coloured by Supplier (vivid, distinct from pastel backgrounds)
Medians : Horizontal tick mark at median per supplier × cheese combo
Handles : 20 K+ points via jitter + transparency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings("ignore")

# ── ① CONFIGURATION  (edit if column names differ) ───────────────────────────
FILE_PATH       = "COA data.xlsx"
SUPPLIER_COL    = None        # None → auto-detect
CHEESE_COL      = None        # None → auto-detect
TEST_COLS       = None        # None → auto-detect first 5 numeric cols

JITTER_STRENGTH = 0.28
POINT_ALPHA     = 0.40
POINT_SIZE      = 16

# ── ② COLOUR PALETTES ─────────────────────────────────────────────────────────
CHEESE_BG = [
    "#D6EAF8","#D5F5E3","#FDEDEC","#FEF9E7","#F4ECF7",
    "#E8F8F5","#FDF2E9","#EBF5FB","#E9F7EF","#FDFEFE",
    "#F9EBEA","#EAF2FF","#E8DAEF","#D1F2EB","#FCF3CF",
    "#FADBD8","#D5D8DC","#D2B4DE","#A9DFBF","#FAD7A0",
    "#A3E4D7","#AED6F1","#F9E79F","#F5CBA7","#D7BDE2",
    "#A9CCE3","#A2D9CE","#F8C471","#82E0AA","#F1948A",
]
SUPPLIER_COLORS = [
    "#E74C3C","#2471A3","#229954","#D35400","#8E44AD",
    "#17A589","#CB4335","#1F618D","#1E8449","#784212",
]

# ── ③ LOAD DATA ───────────────────────────────────────────────────────────────
df = pd.read_excel(FILE_PATH)
print(f"\n✓ Loaded  {len(df):,} rows × {len(df.columns)} columns")
print(f"  Columns: {df.columns.tolist()}\n")

def first_match(df, keywords):
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in str(col).lower():
                return col
    return None

if SUPPLIER_COL is None:
    SUPPLIER_COL = first_match(df, ["supplier","vendor","company","mfr","manufacturer"])
    print(f"  Supplier col : '{SUPPLIER_COL}'")
if CHEESE_COL is None:
    CHEESE_COL = first_match(df, ["cheese","product","type","variety","item","name","sku"])
    print(f"  Cheese col   : '{CHEESE_COL}'")
if TEST_COLS is None:
    exclude = {SUPPLIER_COL, CHEESE_COL}
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]
    TEST_COLS = num_cols[:5]
    print(f"  Test cols    : {TEST_COLS}\n")

for c in TEST_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[SUPPLIER_COL, CHEESE_COL] + TEST_COLS, how="any")
print(f"  Clean rows   : {len(df):,}\n")

# ── ④ ORDERED CATEGORIES ─────────────────────────────────────────────────────
cheese_order    = sorted(df[CHEESE_COL].astype(str).unique())
supplier_order  = sorted(df[SUPPLIER_COL].astype(str).unique())
n_cheese        = len(cheese_order)

cheese_idx      = {c: i for i, c in enumerate(cheese_order)}
cheese_bg_map   = {c: CHEESE_BG[i % len(CHEESE_BG)] for i, c in enumerate(cheese_order)}
sup_color_map   = {s: SUPPLIER_COLORS[i % len(SUPPLIER_COLORS)]
                   for i, s in enumerate(supplier_order)}

df[SUPPLIER_COL] = df[SUPPLIER_COL].astype(str)
df[CHEESE_COL]   = df[CHEESE_COL].astype(str)

# ── ⑤ DRAW ────────────────────────────────────────────────────────────────────
BG = "#12121E"
n_tests  = len(TEST_COLS)
fig_w    = max(24, n_cheese * 0.70)
fig, axes = plt.subplots(n_tests, 1,
                         figsize=(fig_w, 5.2 * n_tests + 1.2),
                         facecolor=BG)
if n_tests == 1:
    axes = [axes]

fig.suptitle("Cheese COA — Test Results by Supplier & Cheese Type",
             fontsize=17, fontweight="bold", color="white", y=1.002)

rng = np.random.default_rng(42)

for tidx, (test_col, ax) in enumerate(zip(TEST_COLS, axes)):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#33334A")

    # background bands
    for i, cheese in enumerate(cheese_order):
        ax.axvspan(i - 0.5, i + 0.5,
                   facecolor=cheese_bg_map[cheese], alpha=0.30, zorder=0)
        if i > 0:
            ax.axvline(i - 0.5, color="#2A2A3C", lw=0.7, zorder=1)

    # scatter + median per supplier
    for sup in supplier_order:
        sc  = sup_color_map[sup]
        sub = df[df[SUPPLIER_COL] == sup]
        xp  = sub[CHEESE_COL].map(cheese_idx).values.astype(float)
        yp  = sub[test_col].values
        xj  = xp + rng.uniform(-JITTER_STRENGTH, JITTER_STRENGTH, len(xp))

        ax.scatter(xj, yp, c=sc, s=POINT_SIZE, alpha=POINT_ALPHA,
                   edgecolors="none", zorder=3,
                   label=sup if tidx == 0 else "_")

        # median tick marks
        for i, cheese in enumerate(cheese_order):
            vals = sub[sub[CHEESE_COL] == cheese][test_col].dropna()
            if len(vals):
                ax.hlines(vals.median(), i - 0.36, i + 0.36,
                          colors=sc, lw=2.4, alpha=0.92, zorder=4)

    # axes formatting
    ax.set_xticks(range(n_cheese))
    short = [str(c)[:16] + "…" if len(str(c)) > 16 else str(c)
             for c in cheese_order]
    ax.set_xticklabels(short, rotation=42, ha="right",
                       fontsize=7.8, color="#BBBBCC")
    ax.set_ylabel(test_col, color="white", fontsize=11, labelpad=8)
    ax.tick_params(axis="y", colors="#AAAACC", labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", color="#222233", lw=0.5, zorder=0)
    ax.set_title(f" Test: {test_col}", loc="left",
                 color="#88CCFF", fontsize=12, pad=5)
    ax.set_xlim(-0.5, n_cheese - 0.5)

    # y-padding
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.05
    ax.set_ylim(ymin - pad, ymax + pad)

# ── ⑥ LEGENDS ────────────────────────────────────────────────────────────────
leg_kw = dict(facecolor="#1E1E30", edgecolor="#44445A",
              labelcolor="white", fontsize=9, framealpha=0.95)

sup_handles = [mpatches.Patch(facecolor=sup_color_map[s], label=s, alpha=0.9)
               for s in supplier_order]

show_n = min(n_cheese, 10)
bg_handles = [mpatches.Patch(facecolor=cheese_bg_map[c], label=c,
                              edgecolor="#666677", linewidth=0.7, alpha=0.75)
              for c in cheese_order[:show_n]]
if n_cheese > show_n:
    bg_handles.append(mpatches.Patch(facecolor="none",
                                     label=f"… +{n_cheese - show_n} more",
                                     edgecolor="none", alpha=0))

l1 = axes[0].legend(handles=sup_handles, title="▌ Supplier",
                    title_fontsize=10,
                    loc="upper right", bbox_to_anchor=(1.0, 1.40),
                    **leg_kw)
l1.get_title().set_color("#88CCFF")

l2 = axes[0].legend(handles=bg_handles,
                    title="▌ Cheese Type (background colour)",
                    title_fontsize=9,
                    loc="upper left", bbox_to_anchor=(0.0, 1.40),
                    ncol=5, **leg_kw)
l2.get_title().set_color("#88CCFF")
axes[0].add_artist(l1)

# caption
fig.text(0.5, -0.005,
         "Each dot = one measurement  |  Horizontal bar = median per supplier × cheese  |  Background colour = cheese type",
         ha="center", fontsize=8.5, color="#888899")

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("viz1_strip_plot.png", dpi=160, bbox_inches="tight", facecolor=BG)
print("✓ Saved → viz1_strip_plot.png")
plt.close()
