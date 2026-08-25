# Modulo Centralizzato delle Funzioni di Loss per l'Addestramento Neurale (loss_functions.py)
# Raccoglie le diverse formulazioni di Loss per il modello PerZoneModel:
#   1. WeightedBCESemanticLoss: BCE pesata con penalizzazione veicoli su Terreno Fuoristrada
#   2. WeightedBCESemanticLoss_penalizzazione_zona_semantica: BCE pesata con penalizzazione estesa a Terreno + Marciapiede
#   3. FocalLoss: Focal Loss dinamica (gamma=2.0) per abbattere i Falsi Positivi su ombre vuote
#   4. FocalLoss_penalizzazione_zona_semantica: Focal Loss dinamica integrata con la penalizzazione veicoli su Terreno + Marciapiede
#   5. AsymmetricLoss: ASL (CVPR 2021) con disaccoppiamento dei gradienti negativi e Margin Shift
#   6. AsymmetricLoss_penalizzazione_zona_semantica: ASL con vincoli neurosimbolici
#   7. FocalLoss_neurosimbolica_completa: Focal Loss con inibizione veicoli e boost VRU
#   8. AsymmetricLoss_neurosimbolica_completa: ASL con inibizione veicoli e boost VRU

# Import di torch per le operazioni sui tensori PyTorch
import torch
# Import di nn (Neural Network Module) per la sottoclasse delle Loss PyTorch
import torch.nn as nn
# Import di F per le funzioni trasversali (binary_cross_entropy_with_logits, sigmoid, ecc.)
import torch.nn.functional as F

class WeightedBCESemanticLoss(nn.Module):
    """
    Funzione di Loss BCE Pesata di Base:
    Penalizza i veicoli predetti sul Terreno Fuoristrada (t_flag).
    """
    def __init__(self, pos_weights=None):
        # Invocazione del costruttore padre
        super(WeightedBCESemanticLoss, self).__init__()
        # Se non vengono forniti pesi positivi, imposta i valori predefiniti per le 6 classi
        if pos_weights is None:
            self.pos_weights = [12.0, 15.0, 20.0, 15.0, 15.0, 12.0]
        else:
            self.pos_weights = pos_weights

    def forward(self, logits, targets, scalars=None):
        # Determina il dispositivo di calcolo (GPU o CPU)
        device = logits.device
        # Calcola la Binary Cross-Entropy senza riduzione
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Converte i pesi delle classi positive in tensore PyTorch
        pos_w_tensor = torch.tensor(self.pos_weights, device=device)
        # Moltiplica i target positivi per i rispettivi pesi di classe
        weights_matrix = torch.ones_like(targets) + targets * (pos_w_tensor - 1.0)
        base_loss = (bce_loss * weights_matrix).mean()
        
        # Se sono fornite le feature scalari semantiche del terreno (almeno 9 scalari)
        if scalars is not None and scalars.shape[1] >= 9:
            probs = torch.sigmoid(logits)
            t_flag = scalars[:, 8:9]  # Terreno Fuoristrada (indice 8)
            s_flag = scalars[:, 5:6]  # Marciapiede (indice 5)
            c_flag = scalars[:, 6:7]  # Strisce pedonali (indice 6)
            
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
    Focal Loss Standard bilanciata con pesi per le classi positive (Lin et al. ICCV 2017).
    Modula dinamicamente il peso dei gradienti in base alla sicurezza della predizione p_t.
    """
    def __init__(self, alpha=0.75, gamma=2.0, pos_weights=[2.0, 3.0, 3.0, 3.0, 3.0, 2.0], reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weights = pos_weights
        self.reduction = reduction

    def forward(self, logits, targets, scalars=None):
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # Calcola p_t (probabilità vera assegnata alla classe di appartenenza)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        # Calcola il fattore di smorzamento dinamico (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma
        
        pos_w = torch.tensor(self.pos_weights, device=logits.device)
        weights_matrix = torch.ones_like(targets) + targets * (pos_w - 1.0)
        loss = weights_matrix * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss_penalizzazione_zona_semantica(nn.Module):
    """
    Focal Loss con Penalizzazione Zona Semantica (Modello Vincente del Progetto per F1-Score):
    Inibisce i veicoli su marciapiede/terreno con peso bilanciato 0.5 per preservare la Recall.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss_penalizzazione_zona_semantica, self).__init__()
        self.focal = FocalLoss(gamma=gamma)

    def forward(self, logits, targets, scalars=None):
        base_focal = self.focal(logits, targets)
        
        if scalars is not None and scalars.shape[1] >= 9:
            probs = torch.sigmoid(logits)
            t_flag = scalars[:, 8:9]  # Terreno Fuoristrada
            s_flag = scalars[:, 5:6]  # Marciapiede
            non_driveable_flag = torch.clamp(t_flag + s_flag, 0.0, 1.0)
            
            # Penalità mitigata a 0.5 per realismo ed evitare la sovra-soppressione dei veicoli
            vehicle_penalty = non_driveable_flag * (1.0 - targets[:, [0, 1, 3]]) * probs[:, [0, 1, 3]] * 0.5
            return base_focal + vehicle_penalty.mean()
            
        return base_focal


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL - Ridnik et al. CVPR 2021) bilanciata per dataset fortemente sbilanciati.
    Separa l'esponente dei positivi (gamma_pos=1.0) da quello dei negativi (gamma_neg=4.0) ed applica il Margin Shift (m=0.05).
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

        # 1. Componente Positiva (y = 1) con esponente gamma_pos=1.0 e pesi pos_weights
        probs_pos = probs
        targets_pos = targets
        loss_pos = targets_pos * torch.log(torch.clamp(probs_pos, min=self.eps))
        if self.gamma_pos > 0:
            loss_pos *= (1.0 - probs_pos) ** self.gamma_pos
            
        pos_w = torch.tensor(self.pos_weights, device=logits.device)
        loss_pos = loss_pos * pos_w

        # 2. Componente Negativa (y = 0) con Margin Shift (clip = 0.05) ed esponente gamma_neg=4.0
        probs_neg = 1.0 - probs
        targets_neg = 1.0 - targets
        
        # Applica lo shift di margine per azzerare i gradienti dei negativi facili con p < 0.05
        if self.clip is not None and self.clip > 0:
            probs_neg = torch.clamp(probs_neg + self.clip, max=1.0)

        loss_neg = targets_neg * torch.log(torch.clamp(probs_neg, min=self.eps))
        if self.gamma_neg > 0:
            loss_neg *= (1.0 - probs_neg) ** self.gamma_neg

        loss = - (loss_pos + loss_neg)
        return loss.mean()


class AsymmetricLoss_penalizzazione_zona_semantica(nn.Module):
    """
    ASL con Penalizzazione Zona Semantica (Modello Vincente del Progetto per Recall Salvavita 100.0% Pedoni):
    Inibisce i veicoli su marciapiede/terreno con peso bilanciato 0.5 per preservare la Recall.
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
            
            # Penalità mitigata a 0.5 per evitare la sovra-soppressione dei veicoli
            vehicle_penalty = non_driveable_flag * (1.0 - targets[:, [0, 1, 3]]) * probs[:, [0, 1, 3]] * 0.5
            return base_asl + vehicle_penalty.mean()
            
        return base_asl


class FocalLoss_neurosimbolica_completa(nn.Module):
    """
    Focal Loss Neurosimbolica Completa:
    Inibisce i veicoli su aree non carrabili (0.5) e promuove VRU su aree pedonali (3.0).
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss_neurosimbolica_completa, self).__init__()
        self.focal = FocalLoss(gamma=gamma)

    def forward(self, logits, targets, scalars=None):
        base_focal = self.focal(logits, targets)
        
        if scalars is not None and scalars.shape[1] >= 9:
            probs = torch.sigmoid(logits)
            t_flag = scalars[:, 8:9]  # Terreno Fuoristrada
            s_flag = scalars[:, 5:6]  # Marciapiede
            c_flag = scalars[:, 6:7]  # Strisce
            
            # 1. Inibizione Veicoli (Auto, Camion, Moto) - Peso mitigato a 0.5
            non_driveable_flag = torch.clamp(t_flag + s_flag, 0.0, 1.0)
            vehicle_penalty = non_driveable_flag * (1.0 - targets[:, [0, 1, 3]]) * probs[:, [0, 1, 3]] * 0.5
            
            # 2. Promozione Pedoni e Biciclette (VRU) su Marciapiede, Strisce e Terreno
            walkway_flag = torch.clamp(s_flag + c_flag + t_flag, 0.0, 1.0)
            ped_bici_boost = walkway_flag * targets[:, [2, 4]] * (1.0 - probs[:, [2, 4]]) * 3.0
            
            return base_focal + vehicle_penalty.mean() + ped_bici_boost.mean()
            
        return base_focal


class AsymmetricLoss_neurosimbolica_completa(nn.Module):
    """
    ASL Neurosimbolica Completa:
    Inibisce i veicoli su aree non carrabili (0.5) e promuove VRU su aree pedonali (3.0).
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super(AsymmetricLoss_neurosimbolica_completa, self).__init__()
        self.asl = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, eps=eps)

    def forward(self, logits, targets, scalars=None):
        base_asl = self.asl(logits, targets)
        
        if scalars is not None and scalars.shape[1] >= 9:
            probs = torch.sigmoid(logits)
            t_flag = scalars[:, 8:9]  # Terreno Fuoristrada
            s_flag = scalars[:, 5:6]  # Marciapiede
            c_flag = scalars[:, 6:7]  # Strisce
            
            # 1. Inibizione Veicoli (Auto, Camion, Moto) - Peso mitigato a 0.5
            non_driveable_flag = torch.clamp(t_flag + s_flag, 0.0, 1.0)
            vehicle_penalty = non_driveable_flag * (1.0 - targets[:, [0, 1, 3]]) * probs[:, [0, 1, 3]] * 0.5
            
            # 2. Promozione Pedoni e Biciclette (VRU) su Marciapiede, Strisce e Terreno
            walkway_flag = torch.clamp(s_flag + c_flag + t_flag, 0.0, 1.0)
            ped_bici_boost = walkway_flag * targets[:, [2, 4]] * (1.0 - probs[:, [2, 4]]) * 3.0
            
            return base_asl + vehicle_penalty.mean() + ped_bici_boost.mean()
            
        return base_asl


# Blocco principale di autoverifica (Self-Test) se eseguito direttamente da riga di comando
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
