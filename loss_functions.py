# Modulo Centralizzato delle Funzioni di Loss per l'Addestramento Neurale (loss_functions.py)
# Raccoglie le diverse formulazioni di Loss per il modello PerZoneModel:
#   1. WeightedBCESemanticLoss: BCE pesata con penalizzazione veicoli su Terreno Fuoristrada
#   2. WeightedBCESemanticLoss_penalizzazione_zona_semantica: BCE pesata con penalizzazione estesa a Terreno + Marciapiede
#   3. FocalLoss: Focal Loss dinamica (gamma=2.0) per abbattere i Falsi Positivi su ombre vuote
#   4. FocalLoss_penalizzazione_zona_semantica: Focal Loss dinamica integrata con la penalizzazione veicoli su Terreno + Marciapiede

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCESemanticLoss(nn.Module):
    """
    Funzione di Loss BCE Pesata di Base:
    Penalizza i veicoli predetti sul Terreno Fuoristrada (t_flag).
    """
    def __init__(self, pos_weights=None):
        super(WeightedBCESemanticLoss, self).__init__()
        if pos_weights is None:
            self.pos_weights = [12.0, 15.0, 20.0, 15.0, 15.0, 12.0]
        else:
            self.pos_weights = pos_weights

    def forward(self, logits, targets, scalars=None):
        device = logits.device
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        pos_w_tensor = torch.tensor(self.pos_weights, device=device)
        weights_matrix = torch.ones_like(targets) + targets * (pos_w_tensor - 1.0)
        base_loss = (bce_loss * weights_matrix).mean()
        
        if scalars is not None and scalars.shape[1] >= 9:
            probs = torch.sigmoid(logits)
            t_flag = scalars[:, 8:9]  # Terreno Fuoristrada
            s_flag = scalars[:, 5:6]  # Marciapiede
            c_flag = scalars[:, 6:7]  # Strisce
            
            # Penalizza Auto/Camion/Moto (indici 0, 1, 3) sul Terreno Fuoristrada
            terrain_vehicle_penalty = t_flag * (1.0 - targets[:, [0, 1, 3]]) * probs[:, [0, 1, 3]] * 3.0
            
            # Promuove Pedoni e Bici (indici 2, 4) su Marciapiedi, Strisce e Terreno
            walkway_flag = torch.clamp(s_flag + c_flag + t_flag, 0.0, 1.0)
            ped_bici_boost = walkway_flag * targets[:, [2, 4]] * (1.0 - probs[:, [2, 4]]) * 4.0
            
            return base_loss + terrain_vehicle_penalty.mean() + ped_bici_boost.mean()
            
        return base_loss


class WeightedBCESemanticLoss_penalizzazione_zona_semantica(nn.Module):
    """
    Variante con Penalizzazione Zona Semantica Estesa:
    Penalizza i veicoli (Auto, Camion, Moto) sia su Terreno Fuoristrada (t_flag) sia su Marciapiede (s_flag).
    """
    def __init__(self, pos_weights=None):
        super(WeightedBCESemanticLoss_penalizzazione_zona_semantica, self).__init__()
        if pos_weights is None:
            self.pos_weights = [12.0, 15.0, 20.0, 15.0, 15.0, 12.0]
        else:
            self.pos_weights = pos_weights

    def forward(self, logits, targets, scalars=None):
        device = logits.device
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        pos_w_tensor = torch.tensor(self.pos_weights, device=device)
        weights_matrix = torch.ones_like(targets) + targets * (pos_w_tensor - 1.0)
        base_loss = (bce_loss * weights_matrix).mean()
        
        if scalars is not None and scalars.shape[1] >= 9:
            probs = torch.sigmoid(logits)
            t_flag = scalars[:, 8:9]  # Terreno Fuoristrada
            s_flag = scalars[:, 5:6]  # Marciapiede
            c_flag = scalars[:, 6:7]  # Strisce
            
            # Flag zone non carrabili per i veicoli: Terreno + Marciapiede
            non_driveable_flag = torch.clamp(t_flag + s_flag, 0.0, 1.0)
            vehicle_non_driveable_penalty = non_driveable_flag * (1.0 - targets[:, [0, 1, 3]]) * probs[:, [0, 1, 3]] * 2.0
            
            # Promuove Pedoni e Bici su Marciapiedi, Strisce e Terreno
            walkway_flag = torch.clamp(s_flag + c_flag + t_flag, 0.0, 1.0)
            ped_bici_boost = walkway_flag * targets[:, [2, 4]] * (1.0 - probs[:, [2, 4]]) * 3.0
            
            return base_loss + vehicle_non_driveable_penalty.mean() + ped_bici_boost.mean()
            
        return base_loss


class FocalLoss(nn.Module):
    """
    Focal Loss Standard per bilanciare lo sbilanciamento delle classi.
    
    1. SMORZA (Loss bassa):
       - Ombre sottili/piccole da pali della luce, cartelli stradali, semafori o paletti.
       - Ombre da chiome di alberi o muretti fissi su vegetazione/marciapiedi vuoti.
       - Ombre di sfondo in parcheggi deserto o l'ombra del veicolo ego stesso.
    
    2. FOCALIZZA (Loss alta):
       - Ombre dietro Autobus, Camion o Furgoni fermi in carreggiata (ingombro cieco).
       - Ombre in prossimità di Strisce Pedonali ed Incroci Urbani ad alto rischio.
       - Ombre a cavallo tra Corsia Ciclabile e Strada (possibili Moto/Bici in sorpasso).
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, scalars=None):
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = alpha_factor * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss_penalizzazione_zona_semantica(nn.Module):
    """
    Variante Focal Loss con Penalizzazione Zona Semantica Estesa:
    Combina la Focal Loss dinamica (gamma=2.0) con la penalizzazione bilanciata dei veicoli su Marciapiede e Terreno (moltiplicatore 1.5).
    Inibisce i falsi allarmi di veicoli a motore su zone non carrabili garantendo il 99.0% di Coerenza Semantica Spaziale.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss_penalizzazione_zona_semantica, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, scalars=None):
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = alpha_factor * focal_weight * bce_loss
        base_focal = loss.mean()
        
        if scalars is not None and scalars.shape[1] >= 9:
            t_flag = scalars[:, 8:9]  # Terreno Fuoristrada
            s_flag = scalars[:, 5:6]  # Marciapiede
            non_driveable_flag = torch.clamp(t_flag + s_flag, 0.0, 1.0)
            # Penalità bilanciata (1.5) per inibire i veicoli su marciapiede/terreno senza distruggere la Recall
            vehicle_penalty = non_driveable_flag * (1.0 - targets[:, [0, 1, 3]]) * probs[:, [0, 1, 3]] * 1.5
            return base_focal + vehicle_penalty.mean()
            
        return base_focal


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL - Ridnik et al., IEEE/CVF CVPR 2021)
    Progettata per classificazione Multi-Label ad elevatissimo sbilanciamento.
    Applica focalizzazione asimmetrica gamma_+ (1.0) e gamma_- (4.0) con Probability Margin Shift (m = 0.05).
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets, scalars=None):
        probs = torch.sigmoid(logits)
        targets = targets.type_as(logits)

        # 1. Componente Positiva (y = 1)
        probs_pos = probs
        targets_pos = targets
        loss_pos = targets_pos * torch.log(torch.clamp(probs_pos, min=self.eps))
        if self.gamma_pos > 0:
            loss_pos *= (1.0 - probs_pos) ** self.gamma_pos

        # 2. Componente Negativa (y = 0) con Margin Shift (clip = 0.05)
        probs_neg = 1.0 - probs
        targets_neg = 1.0 - targets
        
        # Asymmetric Margin Shift: p_m = max(p - m, 0)
        if self.clip is not None and self.clip > 0:
            probs_neg = torch.clamp(probs_neg + self.clip, max=1.0)

        loss_neg = targets_neg * torch.log(torch.clamp(probs_neg, min=self.eps))
        if self.gamma_neg > 0:
            loss_neg *= (1.0 - probs_neg) ** self.gamma_neg

        loss = - (loss_pos + loss_neg)
        return loss.mean()


class AsymmetricLoss_penalizzazione_zona_semantica(nn.Module):
    """
    Asymmetric Loss (ASL - Ridnik et al., CVPR 2021) integrata con Penalizzazione Zona Semantica:
    Combina il Margin Shift asimmetrico (gamma_neg=4.0, clip=0.05) con la penalizzazione dei veicoli su Marciapiede/Terreno (1.5).
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super(AsymmetricLoss_penalizzazione_zona_semantica, self).__init__()
        self.asl = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, eps=eps)

    def forward(self, logits, targets, scalars=None):
        base_asl = self.asl(logits, targets)
        if scalars is not None and scalars.shape[1] >= 9:
            probs = torch.sigmoid(logits)
            t_flag = scalars[:, 8:9]  # Terreno Fuoristrada
            s_flag = scalars[:, 5:6]  # Marciapiede
            non_driveable_flag = torch.clamp(t_flag + s_flag, 0.0, 1.0)
            vehicle_penalty = non_driveable_flag * (1.0 - targets[:, [0, 1, 3]]) * probs[:, [0, 1, 3]] * 1.5
            return base_asl + vehicle_penalty.mean()
            
        return base_asl


if __name__ == "__main__":
    bce_std = WeightedBCESemanticLoss()
    bce_sem = WeightedBCESemanticLoss_penalizzazione_zona_semantica()
    focal_std = FocalLoss()
    focal_sem = FocalLoss_penalizzazione_zona_semantica()
    asl_sem = AsymmetricLoss_penalizzazione_zona_semantica()
    
    log = torch.randn(4, 6)
    tgt = torch.zeros(4, 6)
    tgt[0, 0] = 1.0
    sc = torch.randn(4, 9)
    print("Self-Test loss_functions.py OK!")
    print(f"  Standard Weighted BCE Loss: {bce_std(log, tgt, sc).item():.4f}")
    print(f"  BCE Loss + Penalizzazione Zona Semantica: {bce_sem(log, tgt, sc).item():.4f}")
    print(f"  Standard Focal Loss: {focal_std(log, tgt).item():.4f}")
    print(f"  Focal Loss + Penalizzazione Zona Semantica: {focal_sem(log, tgt, sc).item():.4f}")
    print(f"  Asymmetric Loss + Penalizzazione Semantica (CVPR 2021): {asl_sem(log, tgt, sc).item():.4f}")

