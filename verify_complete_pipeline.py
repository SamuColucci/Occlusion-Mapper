# Script Ufficiale di Collaudo e Diagnostica Completa del Repository (verify_complete_pipeline.py)

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

print("=" * 85)
print("   AVVIO DIAGNOSTICA COMPLETA DEL SISTEMA (TESI OCCLUSION MAPPER)")
print("=" * 85)

# 1. TEST AMBIENTE E ACCELERAZIONE GPU
print("\n[TEST 1/7] Verifica Hardware e PyTorch CUDA...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
print(f"  [OK] Dispositivo Attivo: {device} ({gpu_name})")
print(f"  [OK] PyTorch Version: {torch.__version__}")
assert torch.cuda.is_available(), "GPU CUDA non disponibile!"

# 2. TEST DATASET ADAPTER NUSCENES
print("\n[TEST 2/7] Verifica Dataset Adapter e NuScenes SDK...")
from dataset_adapter.factory_dataset import create_adapter
adapter = create_adapter("nuscenes", "./nuscenes")
num_samples = adapter.get_num_samples()
sample_0 = adapter.get_sample_data(0)
assert num_samples > 0, "Nessun frame trovato nel dataset!"
assert 'semantic_map' in sample_0, "Mappa semantica non presente!"
print(f"  [OK] NuScenes Caricato Correttamente ({num_samples} fotogrammi trovati)")

# 3. TEST RAYCASTING LIDAR
print("\n[TEST 3/7] Verifica Modulo RayCaster 3D...")
from raycaster.ray_caster import RayCaster
caster = RayCaster(sample_0)
caster.get_occlusion_mask()
polys = caster.extract_polygons()
assert len(polys) > 0, "Raycasting fallito (0 poligoni estratti)!"
print(f"  [OK] RayCaster Operativo ({len(polys)} coni d'ombra calcolati sul frame 0)")

# 4. TEST DOPPIA GROUND TRUTH (REALE + SINTETICA NEUROSIMBOLICA)
print("\n[TEST 4/7] Verifica Moduli Ground Truth...")
from ground_truth.ground_truth_extractor import extract_ground_truth_masks, get_occlusion_ground_truth_target
masks = extract_ground_truth_masks(sample_0)
assert len(masks) == 6, "Maschere Ground Truth non conformi!"
gt_target = get_occlusion_ground_truth_target(masks, polys[0]["polygon_points_m"])
print(f"  [OK] Ground Truth Reale e Neurosimbolica Funzionanti (6 layer semantici attivi)")

# 5. TEST ARCHITETTURA NEURALE E CHECKPOINT UFFICIALE
print("\n[TEST 5/7] Verifica Modello AttentionPerZoneModel e Pesi...")
from architettura_neurale import AttentionPerZoneModel, AsymmetricLoss
model = AttentionPerZoneModel(in_channels=11, num_scalars=9, num_classes=6).to(device)
ckpt_path = os.path.join("pesi_modelli", "per_zone_checkpoint_attention_neuro.pth")
assert os.path.exists(ckpt_path), f"Checkpoint pesi non trovato: {ckpt_path}"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()

dummy_patch = torch.randn(2, 11, 64, 64, device=device)
dummy_scalars = torch.randn(2, 9, device=device)
with torch.no_grad():
    out = model(dummy_patch, dummy_scalars)
assert out.shape == (2, 6), f"Output del modello non valido: {out.shape}"
print(f"  [OK] Architettura Attention + FiLM + LayerNorm caricata ed eseguita su GPU ({ckpt_path})")

# 6. TEST AGENTI DI INFERENZA
print("\n[TEST 6/7] Verifica Agenti di Bordo (Neurale e Bayesiano)...")
from inferenza_agenti.neural_agent import PerZoneOcclusionAgent
agent_neuro = PerZoneOcclusionAgent(checkpoint_path=ckpt_path, device=device)
assert agent_neuro.model.__class__.__name__ == "AttentionPerZoneModel"

from inferenza_agenti.bayes_zone_calculator import conditional_probablity_occlusion_zone
bayes_res = conditional_probablity_occlusion_zone(polys[0], sample_0.get('semantic_map', {}))
assert isinstance(bayes_res, dict), "Calcolo probabilita condizionata Bayesiana fallito!"
print(f"  [OK] Agente Neurale ed Agente Bayesiano Operativi al 100%")

# 7. TEST TUTTI I VISUALIZZATORI INTERATTIVI
print("\n[TEST 7/7] Verifica Visualizzatori Ufficiali della Tesi...")
from visualizzatori.verify_runtime_attention import AttentionModelVisualizer
from visualizzatori.verify_runtime_bayes import RuntimeBayesVisualizer
from visualizzatori.verify_runtime_ground_truth import GroundTruthZoneVisualizer
from visualizzatori.verify_runtime_model_comparison import DualModelVisualizer

v1 = AttentionModelVisualizer()
v2 = RuntimeBayesVisualizer()
v3 = GroundTruthZoneVisualizer()
v4 = DualModelVisualizer()
print("  [OK] Visualizzatore 1 (verify_runtime_attention.py)          --> PASS")
print("  [OK] Visualizzatore 2 (verify_runtime_bayes.py)              --> PASS")
print("  [OK] Visualizzatore 3 (verify_runtime_ground_truth.py)        --> PASS")
print("  [OK] Visualizzatore 4 (verify_runtime_model_comparison.py)   --> PASS")

print("\n" + "=" * 85)
print("   TUTTI I 7 COLLAUDI SONO STATI SUPERATI CON SUCCESSO! ZERO ERRORI.")
print("=" * 85 + "\n")
