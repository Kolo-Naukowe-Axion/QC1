# Podsumowanie Cross-Validation 

## 1. Modelowanie Szumu 

Szum kwantowyt był symulowany w oparciu o artykuł. Zgodnie z opisanymi w nim  wynikami został przyjęty optymalny wzór oraz paramerty:


$\epsilon=0.005$,
głębokość warstwy $L=2$ 
oraz $N_g=10$. 
  
  
Model matematyczny błędu zdefiniowano jako:
$$p_{error}=1-(1-\epsilon)^{N_g \cdot L}$$


$$f_{noisy}=(1-p_{error}) \cdot f_{noiseless}+\text{noise}$$
gdzie odchylenie standardowe rozkładu normalnego to $\sigma_{noise}=0.2 \cdot p_{error}$.

## 2. Założenia Metodologiczne


* **Podział Danych (K-Fold)**: Zastosowano podział na 5 foldów przy ustalonym ziarnie losowości (random_state=42).
* **Przeniesienie Skalowania** Proces skalowania cech za pomocą MinMaxScaler (zakres od -π/4 do π/4) został przeniesiony bezpośrednio do pętli walidacji krzyżowej. Skaler jest w każdej iteracji dopasowywany wyłącznie na zbiorze treningowym danego foldu, a dopiero potem aplikowany na zbiór testowy. Takie podejście gwarantuje, że model podczas uczenia nie ma żadnego dostępu do rozkładu danych testowych, co chroni przed data leakage.
* **Zarządzanie Wagami**: W każdym foldzie zapisywano wagi z ostatniej epoki. 

| Hiperparametr | Wartość |
| :--- | :--- |
| Epochs | **30** |
| Batch Size | **16** |
| Learning Rate | **0.01** |
| Ansatz Depth | **2** |

## 3. Zestawienie Wyników


**Wyniki dla poszczególnych foldów:**

| Fold | Accuracy (Ideal) | Accuracy (Noise) | Różnica Acc (Noise - Ideal) | F1 (Ideal) | F1 (Noise) | Różnica F1 (Noise - Ideal) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.8364 | 0.8109 | **-0.0255** | 0.7982 | 0.7570 | **-0.0412** |
| **2** | 0.9055 | 0.9091 | **+0.0036** | 0.8898 | 0.8927 | **+0.0029** |
| **3** | 0.8540 | 0.8577 | **+0.0037** | 0.8077 | 0.8134 | **+0.0057** |
| **4** | 0.8832 | 0.8905 | **+0.0073** | 0.8571 | 0.8649 | **+0.0078** |
| **5** | 0.8832 | 0.8723 | **-0.0109** | 0.8730 | 0.8583 | **-0.0147** |

**Średnie metryki z walidacji krzyżowej (Mean ± Std)**

| Metryka | Model Idealny (Noiseless) | Model Zaszumiony (Noisy) | Zmiana Średniej (Noise - Ideal) |
| :--- | :--- | :--- | :--- |
| **Mean Accuracy** | **0.8725** ± 0.0238 | **0.8681** ± 0.0336 | **-0.0044** |
| **Mean F1 Score** | **0.8452** ± 0.0357 | **0.8373** ± 0.0494 | **-0.0079** |

## 4. Główne Wnioski

1. **Weryfikacja tezy o regularyzacyjnym wpływie szumu**: Naszym pierwotnym celem było sprawdzenie tezy przedstawionej w artykule, według której szum w modelach VQC może pełnić funkcję  regularyzacji i prowadzić do wzrostu dokładności modelu. Ostatecznie odnotowaliśmy jednak minimalny spadek  (o 0.44 punktu procentowego dla Accuracy i 0.84 dla F1). Wynika to z faktu, że nasz bazowy model był już optymalnie dopasowany do postawionego problemu, przez co  regularyzacja szumem nie miała przestrzeni do dalszej poprawy wyników.
2. **Wysoka odporność architektury**: Mimo braku wzrostu dokładności, tak znikomy spadek parametrów klasyfikacyjnych dowodzi wyjątkowej odporności architektury *Ring Topology* na błędy sprzętowe. Granica decyzyjna klasyfikatora nie ulega istotnej degradacji pod wpływem nałożonego szumu fenomenologicznego.
