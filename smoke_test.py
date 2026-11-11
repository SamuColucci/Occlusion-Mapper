"""
smoke_test.py
=============
Script di autodiagnosi rapida (Smoke Test) per verificare che l'ambiente
Conda / venv e tutti i moduli del framework Occlusion-Mapper funzionino a dovere.

Esegue in ~30-60 secondi su 2-3 frame:
 1. Check delle librerie chiave (PyTorch, nuScenes devkit, Shapely, OpenCV).
 2. Check del dataset nuScenes (caricamento metadati e lettura sample 0).
 3. Test RayCasting LiDAR 3D e space-carving sul sample 0 (estrazione poligoni d'ombra).
 4. Caricamento pesi della rete neurale (SE-ResNet con Channel-Attention).
 5. Test Forward Pass su batch sintetico di patch BEV (11 canali, 64x64) + descrittori scalari.
 6. Test istanziazione Neural Agent e pipeline.
"""

import sys
import os
import time

# Aggiungi root al path di Python
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)


def print_step(title):
    print(f"\n{'='*65}\n[TEST] {title}\n{'='*65}")


def main():
    start_total = time.time()
    print("Avvio Smoke Test Occlusion-Mapper (Quick Environment Check)...")

    # ---------------------------------------------------------
    # STEP 1: Import librerie
    # ---------------------------------------------------------
    print_step("1. Verifica Import Librerie Chiave")
    try:
        import torch
        import torchvision
        import numpy as np
        import cv2
        import shapely
        import nuscenes
        print(f"  [OK] PyTorch: {torch.__version__} (CUDA disponibile: {torch.cuda.is_available()})")
        print(f"  [OK] NumPy: {np.__version__}")
        print(f"  [OK] OpenCV: {cv2.__version__}")
        print(f"  [OK] Shapely: {shapely.__version__}")
    except Exception as e:
        print(f"  [FAIL] Errore negli import: {e}")
        return 1

    # ---------------------------------------------------------
    # STEP 2: Dataset Adapter
    # ---------------------------------------------------------
    print_step("2. Caricamento Dataset nuScenes Adapter")
    try:
        from dataset_adapter.factory_dataset import create_adapter
        adapter = create_adapter("nuscenes", "./nuscenes")
        num_samples = len(adapter.all_samples)
        print(f"  [OK] Dataset caricato: {num_samples} sample disponibili.")
        
        sample_0 = adapter.get_sample_data(0)
        s_tok = sample_0["sample_token"]
        lidar_pts = sample_0["points"].shape[0]
        boxes_cnt = len(sample_0["boxes"])
        print(f"  [OK] Sample 0 letto: token={s_tok[:8]}..., punti LiDAR={lidar_pts}, box 3D={boxes_cnt}")
    except Exception as e:
        print(f"  [FAIL] Errore caricamento dataset: {e}")
        return 1

    # ---------------------------------------------------------
    # STEP 3: Raycasting LiDAR 3D su 2 frame
    # ---------------------------------------------------------
    print_step("3. Test RayCasting 3D & Estrazione Zone Occluse (2 frame)")
    try:
        from raycaster.ray_caster import RayCaster
        for idx in [0, 1]:
            s_data = adapter.get_sample_data(idx)
            t0 = time.time()
            caster = RayCaster(s_data)
            caster.get_occlusion_mask()
            polygons = caster.extract_polygons()
            dt = time.time() - t0
            print(f"  [OK] Frame {idx}: Raycasting completato in {dt:.2f}s -> {len(polygons)} zone d'ombra estratte.")
            assert len(polygons) > 0, f"Nessuna zona estratta per frame {idx}!"
    except Exception as e:
        print(f"  [FAIL] Errore RayCasting: {e}")
        return 1

    # ---------------------------------------------------------
    # STEP 4: Caricamento Rete Neurale e Pesi
    # ---------------------------------------------------------
    print_step("4. Caricamento Architettura Neurale e Checkpoint Pesi")
    try:
        from architettura_neurale import AttentionPerZoneModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AttentionPerZoneModel().to(device)
        model.eval()

        ckpt_path = os.path.join(ROOT_DIR, "pesi_modelli", "per_zone_checkpoint_attention_neuro_hybrid.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ROOT_DIR, "pesi_modelli", "per_zone_checkpoint_attention_neuro.pth")
        
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print(f"  [OK] Modello SE-ResNet caricato su {device} dal checkpoint: {os.path.basename(ckpt_path)}")
        else:
            print(f"  [WARN] Nessun checkpoint in pesi_modelli/, uso modello non inizializzato.")
    except Exception as e:
        print(f"  [FAIL] Errore caricamento modello: {e}")
        return 1

    # ---------------------------------------------------------
    # STEP 5: Test Forward Pass Neurale
    # ---------------------------------------------------------
    print_step("5. Test Forward Pass (Patch BEV 11x64x64 + 9 Descrittori Scalari)")
    try:
        # Simulazione batch con 4 zone d'ombra
        batch_size = 4
        dummy_patches = torch.randn(batch_size, 11, 64, 64, device=device)
        dummy_scalars = torch.randn(batch_size, 9, device=device)
        with torch.no_grad():
            outputs = model(dummy_patches, dummy_scalars)
            probs = torch.sigmoid(outputs)
        print(f"  [OK] Forward pass completato! Shape output: {probs.shape} (batch x 6 classi)")
        print(f"  [OK] Esempio probabilita stimate (Zona 0):")
        class_names = ["Auto", "Camion", "Pedone", "Moto", "Bicicletta", "Barriera"]
        for cname, pval in zip(class_names, probs[0].cpu().numpy()):
            print(f"       - {cname:11s}: {pval*100:5.2f}%")
    except Exception as e:
        print(f"  [FAIL] Errore nel forward pass: {e}")
        return 1

    # ---------------------------------------------------------
    # STEP 6: Neural Agent Pipeline
    # ---------------------------------------------------------
    print_step("6. Verifica Neural Agent")
    try:
        from inferenza_agenti.neural_agent import PerZoneOcclusionAgent
        agent = PerZoneOcclusionAgent()
        print(f"  [OK] PerZoneOcclusionAgent inizializzato con successo (Device: {agent.device}).")
    except Exception as e:
        print(f"  [FAIL] Errore Neural Agent: {e}")
        return 1

    # ---------------------------------------------------------
    # ESITO FINALE
    # ---------------------------------------------------------
    elapsed = time.time() - start_total
    print("\n" + "#"*65)
    print(f"  SMOKE TEST SUPERATO CON SUCCESSO! [ALL CHECKS PASSED]")
    print(f"  Tempo totale: {elapsed:.2f} secondi.")
    print(f"  L'ambiente Conda / venv e' perfettamente configurato e operativo.")
    print("#"*65 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
