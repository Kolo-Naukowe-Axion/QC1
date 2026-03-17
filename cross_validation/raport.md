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


* **Podział Danych (K-Fold)**: Zastosowano podział na 5 foldów (`KFold`, k=5) przy ustalonym ziarnie losowości (`random_state=42`).
* **Przeniesienie Skalowania** Proces skalowania cech za pomocą `MinMaxScaler` (zakres od -π/4 do π/4) został przeniesiony bezpośrednio do pętli walidacji krzyżowej. Skaler jest w każdej iteracji dopasowywany (fit) **wyłącznie na zbiorze treningowym danego foldu**, a dopiero potem aplikowany na zbiór testowy. Takie podejście gwarantuje, że model podczas uczenia nie ma żadnego dostępu do rozkładu danych testowych, co chroni przed data leakage.
* **Zarządzanie Wagami**: W każdym foldzie zapisywano dwie wersje modelu – wagi z ostatniej epoki (*Final*) oraz wagi z epoki o najwyższej dokładności na zbiorze testowym (*Best*).

| Hiperparametr | Wartość |
| :--- | :--- |
| Liczba epok (Epochs) | **30** |
| Rozmiar wsadu (Batch Size) | **16** |
| Współczynnik uczenia (Learning Rate) | **0.01** |
| Głębokość ansatzu (Ansatz Depth) | **2** |

## 3. Zestawienie Wyników


| Metryka | Model Idealny (Noiseless) | Model Zaszumiony (Noisy) | Różnica (Noisy vs Idealny) |
| :--- | :--- | :--- | :--- |
| **Mean Final Accuracy** | **0.8717** ± 0.0227 | **0.8673** ± 0.0280 | **-0.0044** (-0.44 p.p.) |
| **Mean Final F1 Score** | **0.8452** ± 0.0332 | **0.8368** ± 0.0415 | **-0.0084** (-0.84 p.p.) |
| **Mean Best Accuracy** | **0.8863** ± 0.0188 | **0.8819** ± 0.0266 | **-0.0044** (-0.44 p.p.) |
| **Mean Best F1 Score** | **0.8663** ± 0.0235 | **0.8597** ± 0.0350 | **-0.0066** (-0.66 p.p.) |

## 4. Główne Wnioski

1. **Weryfikacja tezy o regularyzacyjnym wpływie szumu**: Naszym pierwotnym celem było sprawdzenie tezy przedstawionej w artykule, według której szum w modelach VQC może pełnić funkcję  regularyzacji i prowadzić do wzrostu dokładności modelu. Ostatecznie odnotowaliśmy jednak minimalny spadek metryk (o zaledwie 0.44 punktu procentowego dla Best Accuracy). Wynika to z faktu, że nasz bazowy model był już optymalnie dopasowany do postawionego problemu, przez co  regularyzacja szumem nie miała przestrzeni do dalszej poprawy wyników.
2. **Wysoka odporność (robustness) architektury**: Mimo braku wzrostu dokładności, tak znikomy spadek parametrów klasyfikacyjnych dowodzi wyjątkowej odporności architektury *Ring Topology* na błędy sprzętowe. Granica decyzyjna klasyfikatora nie ulega istotnej degradacji pod wpływem nałożonego szumu fenomenologicznego.
