# **Quantum Banknote Classifier on ODRA 5**

> **Official Research Project:** A hybrid quantum-classical machine learning model designed for **ODRA 5** — Poland's first superconducting quantum computer, launched at **Wrocław University of Science and Technology (WUST)**.

---

## Student Research Group: **KN Axion**
* **Affiliation**: Wrocław University of Science and Technology (**WUST**).
* **Mission**: Exploring practical IT applications of Quantum Information Science.
* **Project Role**: Implementation of a Variational Quantum Classifier (VQC) on the **IQM SPARK Odra 5** infrastructure for banknote authentication.
* **Members**: Iwo Wojtakajtis, Iwo Smura, Rafał Balicki, Karina Leśkiewicz, Maria Płatek, Michał Szczęsny.

---

## Repository Structure
* **`CrossValidation/`** – Scripts ensuring statistical stability and robustness of the results.
* **`evaluation_and_comparison/`** – Detailed analysis including:
    * `ansatz_comparison`: Hardware vs. Simulator performance.
    * `depth_comparison`: Comparizon of circuit depth, gate counts, and execution times across ansatz depths (2, 4, 6) with and without Qiskit transpiler optimization on the IQM ODRA 5 quantum processor.
    * `model_evaluation`: Evaluation of the model.
    * `ram`: RAM memory requirements .
    * `shot_noise`: shot-by-shot noise visualization on IQM SPARK.
* **`models/`** – Implementations of three distinct approaches:
    * `classical_ml`: Standard ML solution used as a performance reference.
    * `model_simulator`: VQC running on a noise-free simulator.
    * `model_odra`: Model specifically tuned for the **IQM Spark (ODRA 5)** hardware.
* **`weights/`** – Pre-trained weights for all models.
* **`eda.ipynb`** – Exploratory Data Analysis of the UCI dataset.

---

## Project Overview
The core objective of this project is to use the **UCI Banknote Authentication** dataset as a benchmark for comparing various machine learning paradigms. 

In this project, we treat the dataset as a foundation for:
* **Ansatz Structure Comparison**: We test defferent circuits (VQC) architectures to determine which gives best results and compare them to classical model.
* **Hardware Constraint Analysis**: The banknote data provides a balanced environment to analyze how gate errors and decoherence on the **ODRA 5** processor affect classification accuracy compared to ideal simulated conditions.


---

## Dataset: Banknote Authentication

The dataset used in this project is sourced from the **UCI Machine Learning Repository**. It was specifically chosen to validate the classification capabilities of the **IQM SPARK ODRA 5** quantum system.



### Data Characteristics:

* **Source**: UCI Banknote Authentication Dataset (ID: 267).

* **Origin**: Data were extracted from images of genuine and forged banknote-like specimens using **Wavelet Transform**.

* **Size**: 1372 instances with 4 continuous features and binary target class.



### Features (Input for Quantum Encoding):

Each of the 4 features is mapped to a single qubit via **Angle Encoding** ($R_y$ gates):

1. **Variance** of Wavelet Transformed image.

2. **Skewness** of Wavelet Transformed image.

3. **Kurtosis** of Wavelet Transformed image.

4. **Entropy** of image.



### Target:

* **Class 0**: Authentic banknote.

* **Class 1**: Counterfeit banknote.



### Automatic Data Loading

The project uses the ucimlrepo library to fetch the data directly from the repository. This ensures the dataset is always available without the need for manual downloads.

```bash

from ucimlrepo import fetch_ucirepo 



# Fetch the banknote authentication dataset 

banknote_authentication = fetch_ucirepo(id=267) 



# Access features (X) and targets (y)

X = banknote_authentication.data.features

y = banknote_authentication.data.targets

```


## Benchmarks & Performance
The framework evaluates five complementary dimensions:
1. compiled resource costs
2. estimated fidelity proxies
3. theoretical expressibility via Kullback-Leibler divergence to the Haar distribution
4. optimisation robustness via five-fold cross-validation under a phenomenological expectation-value noise model
5. end-to-end classification performance directly on the physical QPU using Accuracy and F1 Metrics.
---

## Technical Stack
* **Hardware**: ODRA 5 (Superconducting Qubits, IQM Spark)
* **Software**: Qiskit, PyTorch, Scikit-learn, ucimlrepo
* **Optimization**: Hybrid Quantum-Classical Training (Adam Optimizer via Parameter Shift Rule)

---

## Usage & Workflow
1.  **Exploration**: Start with `eda.ipynb` to understand the input data distribution.
2.  **Architecture**: Review `ansatz.ipynb` to see the circuit tested.
3.  **Comparison**: Check the `evaluation_and_comparison` folder for scripts that contrast results from the simulator and the physical quantum computer.
4.  **Inference**: You can load pre-trained weights from the `/weights` folder to reproduce our results or run inference on your own QPU/Simulator.

---

