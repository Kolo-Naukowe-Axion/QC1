# paper draft — ansatz meta optimization using IQM Spark

### experiments

- **Fidelity calculation:** Compare fidelity from two ansatze: ansatz_odra and ansatz_simulator
    - methodology
        - calculate theoretical final gate error (after transpilation) based on producer’s declarations, compare ansatz_simulator and ansatz_odra
        - note: The results are not related to the technical condition of IQM SPARK
        - transpile: optimization level: 0 (in future research we will look at optimization levels 2 and 3, which are related to technical condition of IQM SPARK)
        - producer’s declaration:
            - single qubit gate fidelity
            typical: ≥ 99.9% (median)
            minimal ≥ 99.7%
            - two - qubit (CZ) fidelity
            typical: ≥ 99.00% (median)
            minimum ≥ 98.00%
            - source: [https://meetiqm.com/products/iqm-spark/#:~:text=Median single-qubit gate fidelity,Minimum%3A ≥ 95%25 Typical%3A](https://meetiqm.com/products/iqm-spark/#:~:text=Median%20single%2Dqubit%20gate%20fidelity,Minimum%3A%20%E2%89%A5%2095%25%20Typical%3A)
        - fidelity calculation (note: it was calculated under the assumption of an independent error model)
            
            $F_{total} = (1 - e_{sq})^{N_{sq}} \cdot (1 - e_{cz})^{N_{cz}}$
            
            - $F_{total}$ – the estimated total circuit fidelity (the overall probability of a successful, error-free execution of the quantum circuit).
            - $e_{sq}$ – the typical (or median) error rate of a single-qubit gate.
            - $N_{sq}$ – the total number of single-qubit gates in the compiled circuit.
            - $e_{cz}$ – the typical (or median) error rate of a two-qubit gate
            - $N_{cz}$– the total number of two-qubit gates in the compiled circuit.
        - random seed = 42
        - depth = 2
    - results
        
        Counts below match transpiled circuits (optimization level 0): **$N_{cz}$** = two-qubit (CZ) gates; **total** = all gates. Single-qubit count is $N_{sq} = \text{total} - N_{cz}$ (used in $F_\text{total}$ above).
        
        | **model** | **typical fidelity** | minimum fidelity | **$N_{cz}$ (two-qubit)** | **total gates** |
        | --- | --- | --- | --- | --- |
        | ansatz_simulator | 60.87% | 31.98% | 35 | 205 |
        | ansatz_odra | 73.76% | 51.47% | 25 | 85 |
    - conclusion
        - ansatz adapted to IQM SPARK has significantly better estimated fidelity (**+12.89 percentage points** typical, **+19.49 pp** minimum) than the non-adapted ansatz under the independent-error model. Higher estimated circuit fidelity implies lower accumulated gate error and shorter runtime on hardware; it **does not by itself** prove better cost-function gradients or avoidance of barren plateaus—those require separate gradient- or landscape analysis—but native-aligned circuits are a standard prerequisite for trainability on noisy devices.

- **Cross-validation: `ansatz_odra` with vs. without phenomenological training noise** (banknote classification)
    - methodology
        - **Data & split.** UCI Banknote Authentication (ID 267); **5-fold** cross-validation with `random_state=42` (see [`cross_validation/`](https://github.com/Kolo-Naukowe-Axion/QC1/tree/main/cross_validation) notebooks, e.g. `CrossValidation_Noise.ipynb`, and `cross_validation/raport.md`).
        - **Preprocessing.** `MinMaxScaler` to **$[-\pi/4,\,\pi/4]$** per fold: **fit on the training fold only**, then transform train and test (no leakage from test distribution).
        - **Quantum model.** Hardware-style **Odra ansatz** (ring CZ connectivity, native-style layers), **5 qubits**, angle-encoded features; `EstimatorQNN` + `StatevectorEstimator`; Pauli-Z readout as in the project notebooks.
        - **Training.** Adam, **learning rate 0.01**, **batch size 16**, **30 epochs** per fold; same optimization settings for noisy and noiseless runs except the noise injection (aligned with `raport.md`).
        - **Noiseless baseline.** Expectation values from the ideal statevector simulator (no phenomenological noise on expectations during training).
        - **Noisy training.** Phenomenological model applied **during training** to expectations: structured error probability **$p_{\text{error}} = 1 - (1-\epsilon)^{N_g L}$** with **$\epsilon = 0.005$**, **$N_g = 10$**, and **$L = 2$** (layer-depth parameter in the hybrid module, as in `raport.md`); noisy expectation **$f_{\text{noisy}} = (1-p_{\text{error}})\, f_{\text{noiseless}} + \text{noise}$** with Gaussian **$\sigma_{\text{noise}} = 0.2\, p_{\text{error}}$**. Inference for reported metrics uses the same ideal estimator so differences isolate **training under noise** vs. **noiseless training** (not a separate hardware deployment).
        - **Depths.** **Depth 2** and **depth 6** refer to the **ansatz depth** hyperparameter (same recipe; deeper circuits use more variational blocks and longer coherent processing in the noiseless channel).
        - **Metrics.** Per fold: accuracy and **macro-F1** on the held-out fold; table entries are **mean ± standard deviation across the five folds** (means shown in the main columns, std in the last two columns).
    - results
    
    
        | model | depth | noise/noiseless | f1 | accuracy | std f1 | std accuracy |
        | --- | --- | --- | --- | --- | --- | --- |
        | ansatz_odra | 2 | noiseless | 0.8452 | 0.8725 | 0.0357 | 0.0238 |
        | ansatz_odra | 2 | noise | 0.8373 | 0.8681 | 0.0494 | 0.0336 |
        | ansatz_odra | 6 | noiseless | 0.8654 | 0.8921 | 0.0337 | 0.0176 |
        | ansatz_odra | 6 | noise | 0.8127 | 0.8499 | 0.0339 | 0.0207 |
    - conclusions
        1. **Depth 2 — small, consistent gap.** Noise-injected training lowers mean F1 by **0.0079** and mean accuracy by **0.0044** relative to the noiseless baseline; cross-fold dispersion (**std**) is slightly **higher** under noise for both metrics. The effect is modest: the decision boundary remains stable for this dataset under the chosen noise strength.
        2. **Depth 6 — larger sensitivity.** The same noise model during training coincides with a **much larger** drop vs. noiseless (**F1 −0.0527**, **accuracy −0.0422**). Deeper variational circuits accumulate more parameterized gates and, in this phenomenological setup, more effective perturbation of gradients and expectations during optimization, so **trainability under noise** becomes the limiting factor before the extra capacity can pay off.
        3. **Connection to recent QML noise analyses.** [Zhu, Dong, Zhang, Li, *Rethinking Quantum Noise in Quantum Machine Learning: When Noise Improves Learning*, arXiv:2601.13275](https://arxiv.org/abs/2601.13275) argue that noise effects are **not uniformly detrimental**: they can behave like an **implicit regularizer** for some initializations while **hurting** already well-optimized models, with a **strong negative correlation** between baseline quality and “noise benefit.” Here, noise **does not** improve held-out metrics; the small depth-2 degradation fits the **“already sufficiently optimized / well-matched task”** regime described in `raport.md`, where little room remains for noise to help. The **depth-6** pattern is closer to **disruption of training** (higher effective error sensitivity) than to beneficial regularization—consistent with the same paper’s emphasis that **noise outcomes depend on architecture depth, initialization, and optimization state**, and that **structure-aware** training—not universal mitigation—is needed.
        4. **Reporting caveat.** Results are **simulator-based** with a **fixed phenomenological noise** recipe; they bound **how much** the chosen VQC can tolerate during training but are not a substitute for **on-device** calibration on IQM Spark (ODRA 5).
- Compare number of gates and depth in transpilation with optimization level = 0 in ansatz_simulator and ansatz_odra in IQM SPARK and state vector estimator (SVE)
    - methodology
        - random seed = 42
        - depth = 2
        - optimization level = 0
    - results
        
        
        | **model** | **environment** | **depth** | **#control gates** | **#total gates** |
        | --- | --- | --- | --- | --- |
        | ansatz_simulator | SVE | 13 | 10 | 20 |
        | ansatz_odra | SVE | 19 | 10 | 30 |
        | ansatz_simulator | IQM | 144 | 35 | 205 |
        | ansatz_odra | IQM | 53 | 25 | 85 |
    - conclusion
        - **Drastic Reduction in Gate Count:** Adapting to native gates reduced the total number of operations from 205 to 85. This directly shortens execution time and reduces cumulative gate error under typical noise models.
        - **Control Layer Optimization:** The number of critical two-qubit (CZ) gates was reduced from 35 to 25, improving the stability of the circuit on the IQM Spark processor.
        - **Robustness through Circuit Shallowness:** Reducing transpiled depth from 144 to 53 improves resilience to decoherence and per-shot error accumulation compared to the simulator-native decomposition.
        - **Hardware Efficiency:** The ansatz_odra is better aligned with the physical native gate set and connectivity of the device.
- Compare two ansatzes (ansatz_odra vs. ansatz_simulator) by how close the **empirical distribution of pairwise state fidelities** is to the **Haar fidelity distribution**, measured as $D_{KL}(P_\text{ansatz}(F),|,P_\text{Haar}(F))$
    - methodology
        - **Parameter sampling.** For each sample, draw independent (\theta^{(a)}, \theta^{(b)} \sim U[0,2\pi]^{n_\text{params}}), build states with `Statevector`, then compute $(F = |\langle \psi(\theta^{(a)}) | \psi(\theta^{(b)}) \rangle|^2).$
        - Repeat **(N)** times (in your CSV: $N \in ({10^4, 2.5\times10^4, 5\times10^4, 7.5\times10^4})$). Use separate RNG streams per ansatz and depth (e.g. `SEED + 100*depth`).
        - **Haar reference.** Theoretical fidelity density for the overlap $f = |\langle\psi_A|\psi_B\rangle|^2$ of two independent Haar-random pure states on $\mathbb{C}^D$ (dimension $D = 2^n$ for $n$ qubits):
            
            $p_\text{Haar}(f) = (D-1)(1-f)^{D-2},\quad f \in [0,1].$
            
        - Discretization: **150 equal bins** on $[0,1]$. Evaluate $(p_\text{Haar})$ at bin midpoints × bin width, then normalize.
        - **KL from histogram.** Histogram of (F) → empirical (P_\text{emp}).
        - Compute $D_{KL}(P_\text{emp},|,P_\text{Haar}^\text{discr.})$ with $\varepsilon=10^{-12}$ smoothing on both distributions (avoids log0).
        
        **Lower KL** means the empirical fidelity distribution is closer (in KL) to the discretized Haar reference.
        
        **Comparing ansatzes.** For each ((N, d)), compare **KL(ODRA)** vs **KL(simulator)**. In your CSV, `gap` = KL(simulator) − KL(ODRA), and `better` is the ansatz with the **smaller** KL.
        
    - results
        
        Table below: **$N = 75\,000$** pairwise fidelity samples per ansatz and depth (see CSV `N_FIDELITY_SAMPLES_PER_DEPTH = 75_000`). Five qubits ($n=5$, $D=2^5$) in `tests/divergence/kl_expressibility_vs_haar.ipynb`.
        
        | depth | KL ansatz_odra | KL ansatz_simulator | KL difference | KL relative difference [%] | winner |
        | --- | --- | --- | --- | --- | --- |
        | 2 | 0.039796 | 0.024453 | -0.015343 | -38.554% | ansatz_simulator |
        | 4 | 0.00074 | 0.000892 | 0.000152 | +20.541% | ansatz_odra |
        | 6 | 0.000324 | 0.000395 | 0.000071 | +21.914% | ansatz_odra |
        | 8 | 0.000343 | 0.000274 | -0.000069 | -20.117% | ansatz_simulator |
    - conclusions
        1. **Shallow depth (depth 2).** Both ansatzes have **large** KL relative to Haar. The **simulator** has a **clearly lower** KL than ODRA at the same (N), and the gap remains **stable** as (N) increases. This suggests the difference is **systematic**, rather than histogram Monte Carlo noise.
            
            **Literature:** the framework of **expressibility** and comparing circuit templates (depth, connectivity, gate types) is given by **Sim, Johnson, Aspuru-Guzik**, *Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms*, [arXiv:1905.10876](https://arxiv.org/abs/1905.10876). They show that **circuit structure** strongly affects expressibility measures, which is consistent with a large gap at small depth.
            
        2. **Larger depths (4–8).** As **depth increases**, both KL values **drop** (distribution of fidelities moves closer to Haar in your discretization), matching the intuition of a **larger-capacity** state family.
            
            **Sim et al.** (same paper) discuss **growth and saturation** of expressibility with circuit complexity — your (d=4)–(8) sit in a regime of **already relatively small** KL compared to (d=2).
            
        3. **Depth ≥ 4: small difference between architectures.** For **(d \in {4,6,8})**, differences between **KL(ODRA)** and **KL(simulator)** are often on the order of **(10^{-3})** down to **(10^{-4})** (at large (N)); which ansatz is “better” at **depth 8** can **flip** across rows — **typical** when the gap is **comparable to estimator noise** (histogram + (\varepsilon)-KL).
            
            **Literature:** using **KL** (and related quantities) to quantify expressibility and **rank** ansatzes, without a single universal “expressive enough” threshold, is common in recent work, e.g. **Zhang, Li, He, Situ**, *Learning the expressibility of quantum circuit ansatz using transformer*, [arXiv:2405.18837](https://arxiv.org/abs/2405.18837) (*Adv. Quantum Technol.* 2025) — KL is treated as a **comparative** tool across circuits, not an absolute cutoff.
            
        4. **Convergence in (N).** For fixed (d), increasing (N) **lowers** both KLs — consistent with **reduced variance** of the histogram; at **depth 2**, KL(ODRA) changes little with (N), indicating a **model–Haar gap** that dominates **Monte Carlo error**.
- create table:
    
    
    | model | environment | f1 | accuracy | std f1 | std accuracy |
    | --- | --- | --- | --- | --- | --- |
    | ansatz_simulator | SVE | 80.72% | 84.36% | - | - |
    | ansatz_odra | SVE | 80.36% | 84.00% | - | - |
    | ansatz_simulator | IQM |  |  |  |  |
    | ansatz_odra | IQM |  |  |  |  |