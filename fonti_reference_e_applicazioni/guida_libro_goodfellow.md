# Verificatore e Confronto Fonti: "Deep Learning" (Goodfellow et al., 2016) vs Codice Sorgente

Questo documento ha lo scopo di **verificare e confrontare direttamente la fonte teorica originale** (il libro di testo *Deep Learning* di Goodfellow, Bengio e Courville - MIT Press, 2016) con il **codice Python/PyTorch reale** implementato nel nostro repository.

---

## 1. Dati Bibliografici Ufficiali della Fonte

* **Titolo**: Deep Learning
* **Autori**: Ian Goodfellow, Yoshua Bengio, Aaron Courville
* **Editore**: MIT Press (2016)
* **Sito Ufficiale**: [deeplearningbook.org](http://www.deeplearningbook.org)

```bibtex
@book{Goodfellow-et-al-2016,
  title={Deep Learning},
  author={Ian Goodfellow and Yoshua Bengio and Aaron Courville},
  publisher={MIT Press},
  year={2016},
  url={http://www.deeplearningbook.org}
}
```

---

## 2. Confronto Sovrapposto: Formule del Libro vs Codice Python

### A. Capitolo 6 (Sezione 6.2.2.2, Pagine 178-180): *Sigmoid Units for Bernoulli Output Distributions*

#### Estratto Originale dal Libro:
> *"A sigmoid output unit is defined by:*
> 
> $$\hat{y} = \sigma(W^T h + b) \quad \text{(Equazione 6.19)}$$
> 
> *where $\sigma$ is the logistic sigmoid function. We can think of the sigmoid output unit as having two components. First, it uses a linear layer to compute $z = W^T h + b$. Next, it uses the sigmoid activation function to convert $z$ into a probability. The $z$ variable defining such a distribution over binary variables is called a logit."*

#### Tabella di Sovrapposizione col Codice:

| Concetto del Libro | Equazione del Libro | Implementazione nel Codice (`architettura_neurale/`) |
| :--- | :--- | :--- |
| **Calcolo Logit $z$** | $z = W^T h + b$ | `logits = self.fc_out(features)` in `per_zone_model.py` |
| **Trasformazione Sigmoide** | $\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$ | `probs = torch.sigmoid(logits)` in `loss_functions.py` |
| **Loss per Bernoulli (BCE)** | $-\log P(y \mid x) = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$ | `F.binary_cross_entropy_with_logits(logits, targets)` |

---

### B. Capitolo 3 (Sezione 3.1, Pagine 51-52): *Incomplete Observability & Bayesian Uncertainty*

#### Estratto Originale dal Libro:
> *"Incomplete observability: Even deterministic systems can appear stochastic when we cannot observe all the variables that drive the behavior of the system."*

#### Tabella di Sovrapposizione col Codice:

| Concetto del Libro | Significato Teorico | Implementazione nel Codice (`inferenza_agenti/`) |
| :--- | :--- | :--- |
| **Incomplete Observability** | Cecità visiva dei sensori dovuta ad ostacoli interposti (coni d'ombra) | Estrazione zone cieche in `dataset_generator_per_zone.py` |
| **Aggiornamento Bayesiano** | $P(Y \mid X) = \frac{P(X \mid Y) P(Y)}{P(X)}$ | Stima della prior e del posterior in `bayes_agent.py` |
| **Tracciamento Temporale** | Integrazione della probabilità nel tempo | Accumulo dinamico probabilistico in `neural_agent_temporal.py` |

---

### C. Capitolo 7 & 8: *Regularization, AdamW & Cosine Scheduler*

#### Estratto Originale dal Libro:
> *"Weight decay is one of the most common forms of regularization (Chapter 7.1.1). Optimization algorithms with adaptive learning rates like Adam adjust the step size for each parameter (Chapter 8.5.3)."*

#### Tabella di Sovrapposizione col Codice:

| Concetto del Libro | Formula / Algoritmo | Implementazione nel Codice (`scratch/train_all_updated_models.py`) |
| :--- | :--- | :--- |
| **$L^2$ Weight Decay** | $\frac{1}{2} \lambda \|w\|_2^2$ con $\lambda = 10^{-4}$ | `torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)` |
| **Cosine LR Scheduler** | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$ | `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)` |
| **Mini-Batch Training** | Mini-batch di dimensione $B$ | `DataLoader(dataset, batch_size=32, shuffle=True)` |

---

## 3. Riepilogo dei 6 Capitoli Utilizzati nel Progetto

1. **Capitolo 3**: Probabilità & Osservabilità Incompleta $\rightarrow$ `bayes_agent.py` e `neural_agent_temporal.py`.
2. **Capitolo 5**: Metriche e iperparametri ($P, T, E$) $\rightarrow$ `evaluate_focal_comparison.py`.
3. **Capitolo 6**: Reti Feedforward, Sigmoide ed Equazioni 6.18-6.23 $\rightarrow$ `per_zone_model.py` e `loss_functions.py`.
4. **Capitolo 7**: Regolarizzazione $L^2$ e Vincoli di Dominio $\rightarrow$ `loss_functions.py` (Loss Neurosimboliche).
5. **Capitolo 8**: Ottimizzazione AdamW e Cosine Annealing $\rightarrow$ `train_all_updated_models.py`.
6. **Capitolo 9**: Reti Convoluzionali 2D su griglie BEV $\rightarrow$ `per_zone_model.py`.
