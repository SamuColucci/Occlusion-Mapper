# Master training script for all 6 updated Focal/ASL models (15 epochs to speed up)
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from addestramento.train_per_zone_focal import train_focal_model
from addestramento.train_per_zone_focal_semantica import train_focal_semantica_model
from addestramento.train_per_zone_focal_completa import train_focal_completa_model
from addestramento.train_per_zone_asl_standard import train_asl_standard_model
from addestramento.train_per_zone_asl import train_asl_model
from addestramento.train_per_zone_asl_completa import train_asl_completa_model

# ponytail: Cache del dataset per evitare di ricaricare nuScenes 6 volte (~200 secondi risparmiati)
from dataset_adapter.dataset_generator_per_zone import OcclusionDatasetPerZone
_cached_ds = None
original_init = OcclusionDatasetPerZone.__init__

def mocked_init(self, *args, **kwargs):
    global _cached_ds
    if _cached_ds is None:
        original_init(self, *args, **kwargs)
        _cached_ds = self
    else:
        self.__dict__.update(_cached_ds.__dict__)

OcclusionDatasetPerZone.__init__ = mocked_init

def main():
    print("=========================================================================")
    print("   AVVIO ADDESTRAMENTO IN SERIE DEI 6 MODELLI AGGIORNATI E BILANCIATI   ")
    print("=========================================================================")
    
    epochs = 15
    
    # 1. Focal Standard
    print("\n[1/6] Addestramento Focal Loss Standard...")
    train_focal_model(epochs=epochs)
    
    # 2. Focal Semantica
    print("\n[2/6] Addestramento Focal Loss Semantica (Inibizione mitigata)...")
    train_focal_semantica_model(epochs=epochs)
    
    # 3. Focal Completa
    print("\n[3/6] Addestramento Focal Loss Completa (Inibizione mitigata + VRU)...")
    train_focal_completa_model(epochs=epochs)
    
    # 4. ASL Standard
    print("\n[4/6] Addestramento ASL Standard...")
    train_asl_standard_model(epochs=epochs)
    
    # 5. ASL Semantica
    print("\n[5/6] Addestramento ASL Semantica (Inibizione mitigata)...")
    train_asl_model(epochs=epochs)
    
    # 6. ASL Completa
    print("\n[6/6] Addestramento ASL Completa (Inibizione mitigata + VRU)...")
    train_asl_completa_model(epochs=epochs)
    
    print("\n=========================================================================")
    print("   ADDESTRAMENTO DI TUTTI E 6 I MODELLI COMPLETATO CON SUCCESSO!         ")
    print("=========================================================================")

if __name__ == "__main__":
    main()
