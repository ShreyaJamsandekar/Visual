"""
VIZ 1 — Strip Plot  |  One chart per Test ID
=============================================
• White background throughout
• One PNG saved per Test ID
• X-axis  : Cheese Types — full column background colour per cheese
• Y-axis  : Test result value
• Dots    : Coloured by Supplier, jittered horizontally
• Median  : Short bar per supplier × cheese combo
• Legend  : RIGHT of chart — never overlaps the plot
• Labels  : Rotated 40°, truncated to avoid overlap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import warnings, os
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  ① CONFIGURATION  — edit column names if auto-detect fails
# ──────────────────────────────────────────────────────────────────────────────
FILE_PATH    = "COA data.xlsx"
SUPPLIER_COL = None     # e.g. "Supplier"
CHEESE_COL   = None     # e.g. "Cheese Type"
TEST_ID_COL  = None     # e.g. "Test ID"   — one chart per unique value here
VALUE_COL    = None     # e.g. "Result"
OUTPUT_DIR   = "."      # where to save the PNGs

JITTER     = 0.28
DOT_SIZE   = 18
DOT_ALPHA  = 0.48

# ──────────────────────────────────────────────────────────────────────────────
#  ② PALETTES
# ──────────────────────────────────────────────────────────────────────────────
CHEESE_BG_PALETTE = [
    "#D6EAF8","#D5F5E3","#FDEDEC","#FEF9E7","#F4ECF7",
    "#E8F8F5","#FDF2E9","#E8DAEF","#D1F2EB","#FCF3CF",
    "#FADBD8","#EBF5FB","#A9DFBF","#FAD7A0","#A3E4D7",
    "#AED6F1","#F9E79F","#F5CBA7","#D7BDE2","#A9CCE3",
    "#A2D9CE","#F8C471","#82E0AA","#F1948A","#85C1E9",
    "#76D7C4","#F0B27A","#BB8FCE","#73C6B6","#F7DC6F",
]
SUPPLIER_PALETTE = [
    "#C0392B","#1A5276","#1E8449","#784212",
    "#6C3483","#0E6655","#2C3E50","#922B21",
    "#154360","#145A32",
]

# ──────────────────────────────────────────────────────────────────────────────
#  ③ HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def find_col(df, keywords):
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in str(col).lower():
                return col
    return None

def trunc(text, n=13):
    s = str(text)
    return (s[:n] + "…") if len(s) > n else s

# ──────────────────────────────────────────────────────────────────────────────
#  ④ LOAD
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_excel(FILE_PATH)
print(f"\n✓ Loaded  {len(df):,} rows × {len(df.columns)} cols")
print(f"  Columns: {df.columns.tolist()}\n")

if SUPPLIER_COL is None:
    SUPPLIER_COL = find_col(df, ["supplier","vendor","mfr","manufacturer","company"])
if CHEESE_COL is None:
    CHEESE_COL   = find_col(df, ["cheese","product","variety","item","sku"])
if TEST_ID_COL is None:
    TEST_ID_COL  = find_col(df, ["test id","testid","test_id","test name","test","parameter","analyte","attribute"])
if VALUE_COL is None:
    VALUE_COL    = find_col(df, ["result","value","reading","measurement","actual","amount"])
    if VALUE_COL is None:
        exc = {SUPPLIER_COL, CHEESE_COL, TEST_ID_COL}
        nums = [c for c in df.select_dtypes(include=np.number).columns if c not in exc]
        VALUE_COL = nums[0] if nums else None

assert all([SUPPLIER_COL, CHEESE_COL, TEST_ID_COL, VALUE_COL]), \
    "Column detection failed — set column names manually in CONFIGURATION section."

print(f"  Supplier → '{SUPPLIER_COL}'")
print(f"  Cheese   → '{CHEESE_COL}'")
print(f"  Test ID  → '{TEST_ID_COL}'")
print(f"  Value    → '{VALUE_COL}'\n")

df[VALUE_COL]    = pd.to_numeric(df[VALUE_COL], errors="coerce")
df[SUPPLIER_COL] = df[SUPPLIER_COL].astype(str).str.strip()
df[CHEESE_COL]   = df[CHEESE_COL].astype(str).str.strip()
df[TEST_ID_COL]  = df[TEST_ID_COL].astype(str).str.strip()
df = df.dropna(subset=[VALUE_COL])

# ──────────────────────────────────────────────────────────────────────────────
#  ⑤ CATEGORY MAPPINGS
# ──────────────────────────────────────────────────────────────────────────────
cheese_order   = sorted(df[CHEESE_COL].unique())
supplier_order = sorted(df[SUPPLIER_COL].unique())
test_ids       = sorted(df[TEST_ID_COL].unique())
n_cheese       = len(cheese_order)

cheese_x    = {c: i  for i, c in enumerate(cheese_order)}
cheese_bg   = {c: CHEESE_BG_PALETTE[i % len(CHEESE_BG_PALETTE)]
               for i, c in enumerate(cheese_order)}
sup_color   = {s: SUPPLIER_PALETTE[i % len(SUPPLIER_PALETTE)]
               for i, s in enumerate(supplier_order)}

print(f"  Cheese types : {n_cheese}")
print(f"  Suppliers    : {len(supplier_order)}  → {supplier_order}")
print(f"  Test IDs     : {len(test_ids)}  → {test_ids}\n")

# ──────────────────────────────────────────────────────────────────────────────
#  ⑥ DRAW — one chart per Test ID
# ──────────────────────────────────────────────────────────────────────────────
for test_id in test_ids:
    sub = df[df[TEST_ID_COL] == test_id]
    if sub.empty:
        continue

    # ── layout -----------------------------------------------------------
    # Each cheese type gets ~0.75 inches; extra 2.8 inches on the right for legend
    fig_w = max(14, n_cheese * 0.75) + 2.8
    fig_h = 6.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")

    # Reserve right margin so legend never overlaps
    fig.subplots_adjust(left=0.07, right=0.78, top=0.88, bottom=0.20)

    # ── cheese column backgrounds ----------------------------------------
    for i, cheese in enumerate(cheese_order):
        ax.axvspan(i - 0.5, i + 0.5,
                   facecolor=cheese_bg[cheese], alpha=1.0, zorder=0)
        if i > 0:
            ax.axvline(i - 0.5, color="white", lw=1.5, zorder=1)

    # ── dots + median bars -----------------------------------------------
    for sup in supplier_order:
        s_df = sub[sub[SUPPLIER_COL] == sup]
        if s_df.empty:
            continue

        xp = s_df[CHEESE_COL].map(cheese_x).values.astype(float)
        yp = s_df[VALUE_COL].values

        seed = abs(hash(sup)) % (2**31)
        xj   = xp + np.random.default_rng(seed).uniform(-JITTER, JITTER, len(xp))

        ax.scatter(xj, yp,
                   color=sup_color[sup],
                   s=DOT_SIZE, alpha=DOT_ALPHA,
                   edgecolors="none", zorder=3)

        # median tick
        for i, cheese in enumerate(cheese_order):
            vals = s_df[s_df[CHEESE_COL] == cheese][VALUE_COL].dropna()
            if len(vals):
                ax.hlines(vals.median(), i - 0.32, i + 0.32,
                          colors=sup_color[sup], lw=2.6, alpha=0.95, zorder=4)

    # ── x-axis -----------------------------------------------------------
    ax.set_xticks(range(n_cheese))
    ax.set_xticklabels(
        [trunc(c, 13) for c in cheese_order],
        rotation=40, ha="right", fontsize=8.5, color="#333333"
    )
    ax.set_xlim(-0.5, n_cheese - 0.5)
    ax.tick_params(axis="x", length=0, pad=3)

    # ── y-axis -----------------------------------------------------------
    ax.set_ylabel("Result", fontsize=10, color="#444444", labelpad=6)
    ax.tick_params(axis="y", labelsize=9, colors="#555555")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", color="#E8E8E8", lw=0.7, zorder=0)
    ax.grid(axis="y", which="minor", color="#F2F2F2", lw=0.35, zorder=0)

    # ── spines -----------------------------------------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    # ── title ------------------------------------------------------------
    ax.set_title(f"  {test_id}", fontsize=13, fontweight="bold",
                 color="#111111", pad=10, loc="left")

    # ── legend OUTSIDE right edge ----------------------------------------
    handles = [
        mpatches.Patch(facecolor=sup_color[s], label=s,
                       linewidth=0, alpha=0.88)
        for s in supplier_order
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

    # ── save -------------------------------------------------------------
    safe = str(test_id).replace("/","-").replace(" ","_").replace(":","-")
    out  = os.path.join(OUTPUT_DIR, f"viz1_{safe}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  ✓ Saved → {out}")
    plt.close()

print("\nDone — one PNG per Test ID saved.")
