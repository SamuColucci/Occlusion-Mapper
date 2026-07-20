"""
Confronto visivo poligoni vecchio export vs nuovo codice.
Mostra i poligoni sovrapposti per verificare che coprano la stessa zona.
Usa frecce <- -> per navigare tra gli oggetti.
"""
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

# Caricamento dati
old = json.load(open("extracted_occlusions/occlusion_sample_0000_ca9a282c9e77460f8360f564131a8af5.json"))
new = json.load(open("scratch/regression_snapshot/occlusions_ref.json"))

old_map = {o["object_token"]: o for o in old["occlusions"]}
new_map = {o["object_token"]: o for o in new["occlusions"]}

# Solo token in comune (non static per chiarezza)
common = [t for t in old_map if t in new_map]
print(f"{len(common)} oggetti in comune da confrontare. Frecce <- -> per navigare, Q per uscire.")

idx = [0]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor="#1a1a2e")

def draw(i):
    for ax in (ax1, ax2, ax3):
        ax.clear()
        ax.set_facecolor("#0f0f23")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=8)

    token = common[i]
    o = old_map[token]
    n = new_map[token]

    poly_old = np.array(o["polygon_points_m"] + [o["polygon_points_m"][0]])
    poly_new = np.array(n["polygon_points_m"] + [n["polygon_points_m"][0]])

    # Limiti comuni
    all_pts = np.vstack([poly_old, poly_new])
    margin = 2.0
    xmin, xmax = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    ymin, ymax = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin

    # 1. Vecchio
    ax1.fill(poly_old[:, 0], poly_old[:, 1], alpha=0.4, color="#ff6b6b", label="vecchio")
    ax1.plot(poly_old[:, 0], poly_old[:, 1], "o-", color="#ff6b6b", markersize=4, linewidth=1.5)
    ax1.set_title(f"VECCHIO ({len(o['polygon_points_m'])} punti)", color="#ff6b6b", fontsize=12, fontweight="bold")
    ax1.set_xlim(xmin, xmax); ax1.set_ylim(ymin, ymax)

    # 2. Nuovo
    ax2.fill(poly_new[:, 0], poly_new[:, 1], alpha=0.4, color="#4ecdc4", label="nuovo")
    ax2.plot(poly_new[:, 0], poly_new[:, 1], "o-", color="#4ecdc4", markersize=4, linewidth=1.5)
    ax2.set_title(f"NUOVO ({len(n['polygon_points_m'])} punti)", color="#4ecdc4", fontsize=12, fontweight="bold")
    ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax)

    # 3. Sovrapposti
    ax3.fill(poly_old[:, 0], poly_old[:, 1], alpha=0.3, color="#ff6b6b")
    ax3.plot(poly_old[:, 0], poly_old[:, 1], "o-", color="#ff6b6b", markersize=3, linewidth=1, label=f"vecchio ({len(o['polygon_points_m'])}pt)")
    ax3.fill(poly_new[:, 0], poly_new[:, 1], alpha=0.3, color="#4ecdc4")
    ax3.plot(poly_new[:, 0], poly_new[:, 1], "s-", color="#4ecdc4", markersize=5, linewidth=2, label=f"nuovo ({len(n['polygon_points_m'])}pt)")
    ax3.legend(loc="upper left", fontsize=9, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax3.set_title("SOVRAPPOSTI", color="white", fontsize=12, fontweight="bold")
    ax3.set_xlim(xmin, xmax); ax3.set_ylim(ymin, ymax)

    fig.suptitle(
        f"[{i+1}/{len(common)}]  {o['object_name']}  |  area: {o['area_sqm']} vs {n['area_sqm']} sqm  |  dist: {o['distance_m']}m",
        color="white", fontsize=13, fontweight="bold"
    )
    fig.canvas.draw_idle()

def on_key(event):
    if event.key == "right":
        idx[0] = (idx[0] + 1) % len(common)
        draw(idx[0])
    elif event.key == "left":
        idx[0] = (idx[0] - 1) % len(common)
        draw(idx[0])
    elif event.key == "q":
        plt.close()

fig.canvas.mpl_connect("key_press_event", on_key)
draw(0)
plt.tight_layout()
plt.show()
