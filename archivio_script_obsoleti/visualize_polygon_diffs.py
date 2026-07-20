"""
visualize_polygon_diffs.py — Visualizza le differenze dei poligoni tra 
i vecchi JSON (extracted_occlusions_old) e i nuovi (extracted_occlusions).

Uso:
  python visualize_polygon_diffs.py --sample-idx 0

Controlli:
  Frecce <- -> : Naviga tra gli oggetti del sample corrente
  Tasto 'n'    : Passa al sample successivo
  Tasto 'p'    : Torna al sample precedente
  Tasto 'q'    : Chiudi
"""
import argparse
import glob
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np


def load_sample_data(sample_idx, old_dir, new_dir):
    # Cerca il file del sample
    old_pattern = os.path.join(old_dir, f"occlusion_sample_{sample_idx:04d}_*.json")
    old_files = glob.glob(old_pattern)
    if not old_files:
        return None, None, f"Nessun file vecchio per il sample {sample_idx}"

    old_file = old_files[0]
    filename = os.path.basename(old_file)
    new_file = os.path.join(new_dir, filename)

    if not os.path.exists(new_file):
        # Prova a cercare solo per idx se il token è cambiato
        new_pattern = os.path.join(new_dir, f"occlusion_sample_{sample_idx:04d}_*.json")
        new_files = glob.glob(new_pattern)
        if new_files:
            new_file = new_files[0]
        else:
            return None, None, f"Nessun file nuovo corrispondente per il sample {sample_idx}"

    try:
        with open(old_file) as f:
            old_data = json.load(f)
        with open(new_file) as f:
            new_data = json.load(f)
        return old_data, new_data, None
    except Exception as e:
        return None, None, str(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-idx", type=int, default=0, help="Indice iniziale del sample")
    p.add_argument("--old-dir", default="extracted_occlusions_old")
    p.add_argument("--new-dir", default="extracted_occlusions")
    args = p.parse_args()

    current_sample = [args.sample_idx]
    current_obj_idx = [0]
    common_tokens = []
    old_map, new_map = {}, {}
    sample_name = [""]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor="#1a1a2e")

    def update_data():
        old_data, new_data, err = load_sample_data(current_sample[0], args.old_dir, args.new_dir)
        if err:
            print(f"[WARN] Impossibile caricare sample {current_sample[0]}: {err}")
            # Pulisce i grafici e mostra errore
            for ax in (ax1, ax2, ax3):
                ax.clear()
                ax.set_facecolor("#0f0f23")
                ax.text(0.5, 0.5, f"Errore: {err}", color="red", ha="center", va="center")
            fig.suptitle(f"Sample {current_sample[0]} - Errore", color="white", fontsize=12)
            fig.canvas.draw_idle()
            return False

        nonlocal common_tokens, old_map, new_map
        old_map = {o["object_token"]: o for o in old_data.get("occlusions", [])}
        new_map = {o["object_token"]: o for o in new_data.get("occlusions", [])}
        common_tokens = [t for t in old_map if t in new_map]
        
        sample_name[0] = f"Sample {current_sample[0]:04d}"
        current_obj_idx[0] = 0
        return True

    def draw():
        if not common_tokens:
            for ax in (ax1, ax2, ax3):
                ax.clear()
                ax.set_facecolor("#0f0f23")
                ax.text(0.5, 0.5, "Nessun oggetto in comune", color="white", ha="center", va="center")
            fig.suptitle(f"{sample_name[0]} - Nessun oggetto", color="white")
            fig.canvas.draw_idle()
            return

        for ax in (ax1, ax2, ax3):
            ax.clear()
            ax.set_facecolor("#0f0f23")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.15, color="white")
            ax.tick_params(colors="white", labelsize=8)

        token = common_tokens[current_obj_idx[0]]
        o = old_map[token]
        n = new_map[token]

        poly_old = np.array(o["polygon_points_m"] + [o["polygon_points_m"][0]])
        poly_new = np.array(n["polygon_points_m"] + [n["polygon_points_m"][0]])

        all_pts = np.vstack([poly_old, poly_new])
        margin = 2.0
        xmin, xmax = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
        ymin, ymax = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin

        # 1. Vecchio
        ax1.fill(poly_old[:, 0], poly_old[:, 1], alpha=0.4, color="#ff6b6b")
        ax1.plot(poly_old[:, 0], poly_old[:, 1], "o-", color="#ff6b6b", markersize=4, linewidth=1.5)
        ax1.set_title(f"VECCHIO ({len(o['polygon_points_m'])} punti)", color="#ff6b6b", fontsize=12, fontweight="bold")
        ax1.set_xlim(xmin, xmax); ax1.set_ylim(ymin, ymax)

        # 2. Nuovo
        ax2.fill(poly_new[:, 0], poly_new[:, 1], alpha=0.4, color="#4ecdc4")
        ax2.plot(poly_new[:, 0], poly_new[:, 1], "o-", color="#4ecdc4", markersize=4, linewidth=1.5)
        ax2.set_title(f"NUOVO ({len(n['polygon_points_m'])} punti)", color="#4ecdc4", fontsize=12, fontweight="bold")
        ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax)

        # 3. Sovrapposti
        ax3.fill(poly_old[:, 0], poly_old[:, 1], alpha=0.3, color="#ff6b6b")
        ax3.plot(poly_old[:, 0], poly_old[:, 1], "o-", color="#ff6b6b", markersize=3, linewidth=1, label="vecchio")
        ax3.fill(poly_new[:, 0], poly_new[:, 1], alpha=0.3, color="#4ecdc4")
        ax3.plot(poly_new[:, 0], poly_new[:, 1], "s-", color="#4ecdc4", markersize=5, linewidth=2, label="nuovo")
        ax3.legend(loc="upper left", fontsize=9, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
        ax3.set_title("SOVRAPPOSTI", color="white", fontsize=12, fontweight="bold")
        ax3.set_xlim(xmin, xmax); ax3.set_ylim(ymin, ymax)

        fig.suptitle(
            f"{sample_name[0]} | Oggetto [{current_obj_idx[0]+1}/{len(common_tokens)}]  {o['object_name']}\n"
            f"Area: {o['area_sqm']} sqm | Dist: {o['distance_m']}m",
            color="white", fontsize=12, fontweight="bold"
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            if common_tokens:
                current_obj_idx[0] = (current_obj_idx[0] + 1) % len(common_tokens)
                draw()
        elif event.key == "left":
            if common_tokens:
                current_obj_idx[0] = (current_obj_idx[0] - 1) % len(common_tokens)
                draw()
        elif event.key == "n":
            current_sample[0] += 1
            update_data()
            draw()
        elif event.key == "p":
            if current_sample[0] > 0:
                current_sample[0] -= 1
                update_data()
                draw()
        elif event.key == "q":
            plt.close()

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Controlli finestra grafica:")
    print("  <-  ->   : Cambia oggetto nel sample")
    print("  tasto 'n': Sample successivo")
    print("  tasto 'p': Sample precedente")
    print("  tasto 'q': Esci")
    
    if update_data():
        draw()
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
