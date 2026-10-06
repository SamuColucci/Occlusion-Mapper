# Modulo Ufficiale della Funzione di Loss per l'Addestramento Neurale (loss_functions.py)
# Implementa la Asymmetric Loss (ASL - Ridnik et al., ICCV 2021) ottimizzata per la predizione multi-label
# fortemente sbilanciata nelle zone d'ombra LiDAR.
#
# Rispetto alla BCE e alla Focal Loss classica:
# - Disaccoppia gli esponenti di focalizzazione (gamma_pos=0.5 o 1.0 vs gamma_neg=4.0)
# - Applica il Probability Margin Shifting (clip=0.05) per azzerare esattamente il gradiente dei negativi facili

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL - Ridnik et al., ICCV 2021) bilanciata per dataset fortemente sbilanciati.
    Separa l'esponente dei positivi (gamma_pos=0.5/1.0) da quello dei negativi (gamma_neg=4.0) ed applica il Margin Shift (m=0.05).
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, pos_weights=[2.0, 3.0, 3.0, 3.0, 3.0, 2.0], eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.pos_weights = pos_weights
        self.eps = eps

    def forward(self, logits, targets, scalars=None):
        probs = torch.sigmoid(logits)
        targets = targets.type_as(logits)

        # 1. Componente Positiva (y = 1) con modulazione gamma_pos e pos_weights
        probs_pos = probs
        targets_pos = targets
        loss_pos = targets_pos * torch.log(torch.clamp(probs_pos, min=self.eps))
        if self.gamma_pos > 0:
            loss_pos *= torch.clamp(1.0 - probs_pos, min=1e-6) ** self.gamma_pos

        pos_w = torch.tensor(self.pos_weights, device=logits.device)
        loss_pos = loss_pos * pos_w

        # 2. Componente Negativa (y = 0) con Margin Shift (clip) e modulazione gamma_neg
        probs_neg = 1.0 - probs
        targets_neg = 1.0 - targets

        if self.clip is not None and self.clip > 0:
            probs_neg = torch.clamp(probs_neg + self.clip, max=1.0)

        loss_neg = targets_neg * torch.log(torch.clamp(probs_neg, min=self.eps))
        if self.gamma_neg > 0:
            loss_neg *= torch.clamp(1.0 - probs_neg, min=1e-6) ** self.gamma_neg

        # Se una classe ha pos_weight == 0 (es. Barriere escluse), azzera il contributo
        active_classes = (pos_w > 0).float()
        loss = - (loss_pos + loss_neg) * active_classes
        return loss.mean()


if __name__ == "__main__":
    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.5, clip=0.05, pos_weights=[1.2, 2.2, 2.0, 1.8, 1.8, 0.0])
    log = torch.randn(4, 6)
    tgt = torch.zeros(4, 6)
    tgt[0, 0] = 1.0
    val = asl(log, tgt)
    print(f"[TEST ASL]: Loss calcolata con successo = {val.item():.4f}")
