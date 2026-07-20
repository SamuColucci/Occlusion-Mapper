# Approccio Bayesiano Dinamico con Modulazione Spaziale Semantica e Memoria Temporale

Questo documento descrive in dettaglio il funzionamento dell'**Agente Bayesiano Dinamico Unificato** utilizzato per calcolare la probabilità condizionata della presenza di ostacoli nascosti (occlusioni) in scenari di guida autonoma sul dataset NuScenes.

---

## 1. Architettura dell'Agente

1. **Superficie di Appoggio Piatta (Flat Semantic Layers)**: Intersecazione geometrica spaziale tra il cono d'ombra dell'occlusione e la mappa HD stradale.
2. **Dimensioni Fisiche della Sotto-zona (OBB Width)**: Restrizione geometrica basata sulle dimensioni reali dell'occlusione (una macchina o un camion non possono trovarsi in corsie o corridoi più stretti del proprio ingombro fisico).
3. **Memoria Temporale Causale (Seen Instances)**: Tracciamento cronologico dei soli frame passati per rintracciare veicoli o pedoni precedentemente visibili che potrebbero essersi spostati all'interno della zona d'ombra corrente.

---

## 2. Divisione Geometrica in Sotto-zone (Sub-zones)
L'occlusione non viene analizzata come un unico blocco. Viene ritagliata geometricamente tramite libreria `Shapely` intersecandola con le classi semantiche piane di NuScenes. Ogni occlusione viene così divisa in sotto-zone distinte, ognuna con le proprie coordinate locali, la propria area in $m^2$ e la propria larghezza OBB:

*   **`flat.driveable_surface` (Superficie Carrabile)**: Le strade destinate al transito automobilistico.
*   **`flat.sidewalk` (Marciapiede)**: Aree dedicate ai pedoni.
*   **`flat.other` (Altro Piano / Parcheggi)**: Cortili, piazzali di cemento e parcheggi, dove possono trovarsi sia veicoli che persone.
*   **`ped_crossing` (Strisce Pedonali)**: Attraversamenti pedonali ad alto rischio per pedoni e velocipedi.
*   **`flat.terrain` (Terreno / Prato)**: Aree sterrate, collinette ed erba (esclude i veicoli e favorisce i pedoni).

---

## 3. Calcolo della Probabilità Condizionata
Per ciascuna sotto-zona, la probabilità per le varie classi di target (10 classi di NuScenes) viene determinata come:

$$P(\text{Classe} \mid \text{Superficie}, \text{Larghezza}) = P_{\text{base}}(\text{Classe} \mid \text{Superficie}) \times K_{\text{width}}(\text{Larghezza})$$

### 3.1. Probabilità Semantica di Base ($P_{\text{base}}$)
Rappresenta la probabilità intrinseca di trovare un oggetto su quella superficie pura:
*   Un **Pedone** ha massima probabilità su `sidewalk` (35%) e `ped_crossing` (60%).
*   Un'**Auto** ha massima probabilità su `other_flat` (65%) e `driveable_surface` (30%).
*   Un **Camion** o **Bus** ha probabilità 0% su `sidewalk` e `terrain`.

### 3.2. Coefficiente di Inibizione Fisica ($K_{\text{width}}$)
Un filtro geometrico che inibisce la probabilità degli oggetti voluminosi se la larghezza OBB del cono d'ombra ($W$) è troppo piccola:
*   Se $W < 1.0\text{ m}$: $K_{\text{width}}(\text{Auto}) = 0.0$, $K_{\text{width}}(\text{Camion}) = 0.0$, $K_{\text{width}}(\text{Pedone}) = 1.0$ (le auto non possono passare).
*   Se $1.0\text{ m} \le W < 1.8\text{ m}$: $K_{\text{width}}(\text{Auto}) = 0.05$ (solo auto piccolissime), $K_{\text{width}}(\text{Camion}) = 0.0$.
*   Se $W \ge 4.0\text{ m}$: $K_{\text{width}}(\text{Camion}) = 1.0$, $K_{\text{width}}(\text{Auto}) = 1.0$, $K_{\text{width}}(\text{Pedone}) = 0.60$ (la strada larga diminuisce la probabilità relativa dei pedoni a favore dei veicoli).

---

## 4. Integrazione della Memoria Storica
Se nei frame passati della scena corrente (senza mai guardare al futuro, rispettando il principio di causalità a runtime) un oggetto precedentemente visibile scompare all'interno del cono di espansione dell'occlusione (con buffer di tolleranza di $0.2\text{ metri}$):
*   La probabilità condizionata per quella specifica classe nella sub-zona interessata riceve un **boost immediato al 95%** ($0.95$), ad indicare l'elevata probabilità che l'oggetto stia occupando la zona d'ombra.

---

## 5. Visualizzazione HUD e Legenda
Il visualizzatore mostra:
1. **HUD a sinistra**: Le percentuali fisiche esatte delle 5 superfici di appoggio per la zona d'ombra selezionata, e le probabilità stimate.
2. **Clutter-Free Categoria**: Vengono stampate sempre le 4 classi principali (`Auto`, `Pedone`, `Camion`, `Bicicletta`). Le altre 6 classi secondarie di NuScenes (`Moto`, `Bus`, `Rimorchio`, `Barriera`, `Cono`, `Altro`) vengono disegnate solo se la loro probabilità stimata supera lo **0.5%**, eliminando le righe a 0% che causerebbero rumore visivo.
3. **Mappa a destra**: La ricostruzione stradale HD, i punti LiDAR, le sub-zone colorate con etichetta centrata al centroide con la percentuale di appoggio.
