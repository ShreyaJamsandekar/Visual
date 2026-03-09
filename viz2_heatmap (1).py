"""
VIZ 2 — Heatmap  |  One chart per Test ID
==========================================
• White background
• One PNG per Test ID
• Rows    : Cheese Types
• Columns : Suppliers
• Colour  : Mean result value — diverging scale centred at the overall mean
• Labels  : Raw mean shown inside each cell (hidden if too small to read)
• Legend  : Colorbar placed BELOW the chart — never overlaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
#  ② HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def find_col(df, kws):
    for kw in kws:
        for col in df.columns:
            if kw.lower() in str(col).lower():
                return col
    return None

def trunc(text, n=18):
    s = str(text)
    return (s[:n] + "…") if len(s) > n else s

# ──────────────────────────────────────────────────────────────────────────────
#  ③ LOAD
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
        exc = {SUPPLIER_COL, CHEESE_COL, TEST_ID_COL}
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

print(f"  Supplier col : '{SUPPLIER_COL}'  |  {n_sup} suppliers")
print(f"  Cheese col   : '{CHEESE_COL}'    |  {n_cheese} types")
print(f"  Test IDs     : {len(test_ids)}")

# ──────────────────────────────────────────────────────────────────────────────
#  ④ DRAW — one chart per Test ID
# ──────────────────────────────────────────────────────────────────────────────
for test_id in test_ids:
    sub = df[df[TEST_ID_COL] == test_id]
    if sub.empty:
        continue

    # pivot: rows = cheese, cols = supplier, values = mean result
    pivot = (sub.groupby([CHEESE_COL, SUPPLIER_COL])[VALUE_COL]
               .mean()
               .unstack(SUPPLIER_COL)
               .reindex(index=cheese_order, columns=supplier_order))

    mat    = pivot.values.astype(float)
    global_mean = np.nanmean(mat)
    half_range  = max(abs(np.nanmax(mat) - global_mean),
                      abs(global_mean - np.nanmin(mat))) * 1.05
    norm = mcolors.TwoSlopeNorm(
        vmin=global_mean - half_range,
        vcenter=global_mean,
        vmax=global_mean + half_range,
    )

    # ── figure geometry ---------------------------------------------------
    cell_w = max(1.4, 10 / n_sup)
    cell_h = max(0.40, 16 / n_cheese)
    fig_w  = n_sup * cell_w + 4.0
    fig_h  = n_cheese * cell_h + 2.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.22, right=0.92, top=0.88, bottom=0.18)

    im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r", norm=norm)

    # white grid lines between cells
    for x in np.arange(-0.5, n_sup, 1):
        ax.axvline(x, color="white", lw=1.5)
    for y in np.arange(-0.5, n_cheese, 1):
        ax.axhline(y, color="white", lw=0.8)

    # cell annotations — only if cells are large enough to read
    min_cell_px = cell_h * 72 * 0.8   # rough pixel height
    annotate    = min_cell_px > 14

    for r in range(n_cheese):
        for c in range(n_sup):
            v = mat[r, c]
            if np.isnan(v):
                ax.text(c, r, "–", ha="center", va="center",
                        fontsize=8, color="#AAAAAA")
                continue
            if annotate:
                # choose black or white text based on background brightness
                z = norm(v)
                txt_col = "white" if (z < 0.3 or z > 0.7) else "#222222"
                ax.text(c, r, f"{v:.2f}",
                        ha="center", va="center",
                        fontsize=max(6, min(9, 70 / n_cheese)),
                        color=txt_col)

    # ── axes labels -------------------------------------------------------
    ax.set_xticks(range(n_sup))
    ax.set_xticklabels([trunc(s, 12) for s in supplier_order],
                       rotation=30, ha="right", fontsize=9, color="#333333")
    ax.set_yticks(range(n_cheese))
    ax.set_yticklabels([trunc(c, 20) for c in cheese_order],
                       fontsize=max(6.5, min(9, 180 / n_cheese)), color="#333333")
    ax.tick_params(length=0)

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title(f"  {test_id}  —  Mean Result by Supplier & Cheese Type",
                 fontsize=12, fontweight="bold", color="#111111",
                 pad=10, loc="left")

    # ── colorbar BELOW the chart -----------------------------------------
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        fraction=0.035, pad=0.12, aspect=40)
    cbar.ax.tick_params(labelsize=8, colors="#555555")
    cbar.set_label("Mean result value", fontsize=8.5, color="#444444")
    cbar.outline.set_edgecolor("#CCCCCC")

    # ── save -------------------------------------------------------------
    safe = str(test_id).replace("/","-").replace(" ","_").replace(":","-")
    out  = os.path.join(OUTPUT_DIR, f"viz2_{safe}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  ✓ Saved → {out}")
    plt.close()

print("\nDone — one heatmap per Test ID saved.")
